"""Inspect execution-role trust without impersonating the SageMaker service."""

import json
from fnmatch import fnmatchcase
from typing import Any, Optional
from urllib.parse import unquote

from hf_cloud.core.exceptions import ProviderError

SAGEMAKER_PRINCIPAL = "sagemaker.amazonaws.com"


def _values(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def _principal_matches(principal: Any) -> bool:
    return principal == "*" or (
        isinstance(principal, dict)
        and (SAGEMAKER_PRINCIPAL in _values(principal.get("Service")) or "*" in _values(principal.get("AWS")))
    )


def _applies_to_sagemaker(statement: dict) -> bool:
    principal = _principal_matches(statement.get("Principal"))
    if "NotPrincipal" in statement:
        principal = not _principal_matches(statement["NotPrincipal"])
    actions = statement.get("Action", statement.get("NotAction", []))
    action = any(fnmatchcase("sts:assumerole", pattern.lower()) for pattern in _values(actions))
    if "NotAction" in statement:
        action = not action
    return principal and action


def _any_match(matches: list[Optional[bool]]) -> Optional[bool]:
    if True in matches:
        return True
    return None if None in matches else False


def _condition_matches(operator: str, key: str, values: Any, arn: str, region: str) -> Optional[bool]:
    partition, account = arn.split(":")[1], arn.split(":")[4]
    key = key.lower()
    if operator == "Bool" and key == "aws:principalisawsservice":
        return any(str(value).lower() == "true" for value in _values(values))
    operator = operator.removesuffix("IfExists")
    negative = operator in {"StringNotEquals", "StringNotLike", "ArnNotEquals", "ArnNotLike"}
    positive = operator.replace("Not", "")
    if positive not in {"StringEquals", "StringLike", "ArnEquals", "ArnLike"}:
        return None
    matches = []
    for value in _values(values):
        if key == "aws:sourceaccount":
            matches.append(fnmatchcase(account, value) if positive.endswith("Like") else account == value)
        elif key == "aws:sourcearn":
            parts = value.split(":", 5)
            context = ["arn", partition, "sagemaker", region, account]
            compare = (lambda actual, pattern: actual == pattern) if positive == "StringEquals" else fnmatchcase
            if len(parts) != 6 or not all(compare(actual, pattern) for actual, pattern in zip(context, parts)):
                matches.append(False)
            elif positive == "StringEquals":
                matches.append(False if "*" in parts[5] or "?" in parts[5] else None)
            else:
                # ModelBuilder chooses model resource names later. Only a full
                # resource wildcard can be confirmed before creating the model.
                matches.append(True if parts[5] == "*" else None)
        else:
            matches.append(None)
    result = _any_match(matches)
    return not result if negative and result is not None else result


def _conditions_match(conditions: dict, arn: str, region: str) -> Optional[bool]:
    results = [
        _condition_matches(operator, key, values, arn, region)
        for operator, entries in conditions.items()
        for key, values in entries.items()
    ]
    if False in results:
        return False
    return None if None in results else True


def validate_role_trust(role: dict, region: str) -> bool:
    """Reject definite trust failures; return False for conditions only AWS can evaluate.

    This validates service trust, not iam:PassRole, SCPs, or runtime permissions.
    STS AssumeRole as the caller cannot test assumption as the SageMaker service.
    """
    arn = role["Arn"]
    try:
        policy = role["AssumeRolePolicyDocument"]
        if isinstance(policy, str):
            policy = json.loads(unquote(policy))
        allowed, denied = [], []
        for statement in _values(policy.get("Statement", [])):
            if _applies_to_sagemaker(statement):
                match = _conditions_match(statement.get("Condition", {}), arn, region)
                if statement.get("Effect") == "Allow":
                    allowed.append(match)
                elif statement.get("Effect") == "Deny":
                    denied.append(match)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ProviderError(
            "sagemaker", "Cannot read execution role trust policy", details=f"{arn}: {error}"
        ) from error
    if True in denied or not any(match is not False for match in allowed):
        raise ProviderError(
            "sagemaker",
            "Execution role does not allow SageMaker to assume it",
            details=(
                f"{arn}. Its trust policy must allow sts:AssumeRole for service principal "
                f"{SAGEMAKER_PRINCIPAL}, without a matching Deny and with conditions matching "
                f"account {arn.split(':')[4]} and region {region}. "
                "Choose a SageMaker execution role with --role-name or --role."
            ),
        )
    return True in allowed and None not in denied
