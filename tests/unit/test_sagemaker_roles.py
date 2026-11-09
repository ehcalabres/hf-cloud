"""Role lookup and trust-policy preflight without AWS resources."""

import json
import sys
from types import ModuleType
from unittest.mock import Mock
from urllib.parse import quote

import pytest
from botocore.exceptions import ClientError

from hf_cloud.core.exceptions import ProviderError
from hf_cloud.providers.sagemaker.client import SageMakerClient
from hf_cloud.providers.sagemaker.roles import validate_role_trust

ARN = "arn:aws:iam::123456789012:role/team/MyRole"


def statement(**overrides):
    return {
        "Effect": "Allow",
        "Action": "sts:AssumeRole",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        **overrides,
    }


def role(*statements):
    return {"Arn": ARN, "AssumeRolePolicyDocument": {"Statement": list(statements) or [statement()]}}


@pytest.fixture
def client():
    client = SageMakerClient(region="us-east-1")
    client._iam_client = Mock()
    client.iam.get_role.return_value = {"Role": role()}
    return client


@pytest.mark.parametrize("identifier", ["MyRole", ARN])
def test_name_and_arn_resolve_the_actual_iam_role_once(client, identifier):
    assert client.get_execution_role(identifier) == ARN
    client.iam.get_role.assert_called_once_with(RoleName="MyRole")


def test_auto_discovered_roles_are_also_validated(client, monkeypatch):
    module = ModuleType("sagemaker.core.helper.session_helper")
    module.get_execution_role = Mock(return_value=ARN)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    client.get_sagemaker_session = Mock()
    assert client.get_execution_role() == ARN
    client.iam.get_role.assert_called_once_with(RoleName="MyRole")
    client.iam.get_role.return_value = {
        "Role": role(statement(Principal={"AWS": "arn:aws:iam::123456789012:root"}))
    }
    with pytest.raises(ProviderError, match="does not allow SageMaker"):
        client.get_execution_role()


@pytest.mark.parametrize(
    "code,expected",
    [
        ("NoSuchEntity", "does not exist"),
        ("AccessDenied", "iam:GetRole permission is required"),
        ("ServiceFailure", "Cannot load execution role"),
    ],
)
def test_iam_errors_remain_actionable(client, code, expected):
    client.iam.get_role.side_effect = ClientError({"Error": {"Code": code, "Message": "AWS error"}}, "GetRole")
    with pytest.raises(ProviderError, match=expected):
        client.get_execution_role("MyRole")


@pytest.mark.parametrize(
    "identifier", ["", "a role", "path/Role", "arn:aws:s3:::bucket", "arn:aws:iam::bad:role/Role"]
)
def test_invalid_identifiers_fail_without_iam(client, identifier):
    with pytest.raises(ProviderError):
        client.get_execution_role(identifier)
    client.iam.get_role.assert_not_called()


def test_a_different_account_or_role_path_is_not_silently_substituted(client):
    with pytest.raises(ProviderError, match="does not match"):
        client.get_execution_role(ARN.replace("123456789012", "111111111111"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"Action": "sts:*"},
        {"Action": ["sts:TagSession", "sts:AssumeRole"]},
        {"Principal": {"Service": ["ec2.amazonaws.com", "sagemaker.amazonaws.com"]}},
        {"Principal": "*"},
        {"Principal": {"AWS": "*"}},
        {"Condition": {"StringEquals": {"aws:SourceAccount": "123456789012"}}},
        {"Condition": {"ArnLike": {"aws:SourceArn": "arn:aws:sagemaker:*:123456789012:*"}}},
        {
            "Condition": {
                "ArnLike": {
                    "AWS:SourceArn": [
                        "arn:aws:sagemaker:us-west-2:123456789012:*",
                        "arn:aws:sagemaker:us-east-1:123456789012:*",
                    ]
                }
            }
        },
    ],
)
def test_service_trust_accepts_standard_policies(overrides):
    assert validate_role_trust(role(statement(**overrides)), "us-east-1")


@pytest.mark.parametrize(
    "overrides",
    [
        {"Principal": {"Service": "ec2.amazonaws.com"}},
        {"Principal": {"AWS": "arn:aws:iam::123456789012:root"}},
        {"Action": "sts:AssumeRoleWithSAML"},
        {"Condition": {"StringEquals": {"aws:SourceAccount": "111111111111"}}},
        {"Condition": {"ArnLike": {"aws:SourceArn": "arn:aws:sagemaker:us-west-2:123456789012:*"}}},
        {"Condition": {"ArnLike": {"aws:SourceArn": "arn:aws:sagemaker:*:111111111111:*"}}},
    ],
)
def test_missing_trust_or_incompatible_known_conditions_fail(overrides):
    with pytest.raises(ProviderError, match="does not allow SageMaker to assume it"):
        validate_role_trust(role(statement(**overrides)), "us-east-1")


def test_explicit_deny_overrides_allow():
    with pytest.raises(ProviderError, match="does not allow SageMaker"):
        validate_role_trust(role(statement(), statement(Effect="Deny", Action="*", Principal="*")), "us-east-1")


def test_wildcard_aws_principal_deny_is_not_ignored():
    with pytest.raises(ProviderError, match="does not allow SageMaker"):
        validate_role_trust(role(statement(), statement(Effect="Deny", Principal={"AWS": "*"})), "us-east-1")


def test_notaction_and_notprincipal_are_respected():
    allow = statement()
    del allow["Action"]
    allow["NotAction"] = "sts:AssumeRoleWithSAML"
    assert validate_role_trust(role(allow), "us-east-1")
    allow["NotAction"] = "sts:AssumeRole"
    with pytest.raises(ProviderError):
        validate_role_trust(role(allow), "us-east-1")
    deny = statement(Effect="Deny")
    del deny["Principal"]
    deny["NotPrincipal"] = {"Service": "sagemaker.amazonaws.com"}
    assert validate_role_trust(role(statement(), deny), "us-east-1")


def test_nonmatching_conditional_deny_does_not_override_allow():
    assert validate_role_trust(
        role(
            statement(),
            statement(
                Effect="Deny",
                Condition={"StringNotEquals": {"aws:SourceAccount": "123456789012"}},
            ),
        ),
        "us-east-1",
    )


def test_unknown_conditions_are_deferred_and_announced(client, capsys):
    client.iam.get_role.return_value = {
        "Role": role(statement(Condition={"StringEquals": {"aws:SourceOrgID": "o-example"}}))
    }
    assert client.get_execution_role("MyRole") == ARN
    assert "require deployment context" in capsys.readouterr().out


def test_resource_specific_conditions_do_not_claim_to_be_verified():
    assert not validate_role_trust(
        role(
            statement(
                Condition={
                    "ArnLike": {"aws:SourceArn": "arn:aws:sagemaker:us-east-1:123456789012:model/my-model*"}
                },
            )
        ),
        "us-east-1",
    )


def test_unknown_deny_is_not_overlooked():
    assert not validate_role_trust(
        role(
            statement(),
            statement(
                Effect="Deny",
                Condition={"StringEquals": {"aws:SourceOrgID": "o-example"}},
            ),
        ),
        "us-east-1",
    )


def test_encoded_policy_and_single_statement_are_supported():
    data = role()
    data["AssumeRolePolicyDocument"] = quote(json.dumps({"Statement": statement()}))
    assert validate_role_trust(data, "us-east-1")


@pytest.mark.parametrize("policy", [None, "bad JSON", {"Statement": "bad statement"}, {"Statement": []}])
def test_unreadable_or_empty_policy_is_not_accepted(policy):
    data = role()
    data["AssumeRolePolicyDocument"] = policy
    with pytest.raises(ProviderError):
        validate_role_trust(data, "us-east-1")
