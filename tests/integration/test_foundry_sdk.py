"""Exercise the real preview SDK against an in-memory ARM transport, never Azure."""

import json
import time

import pytest

pytest.importorskip("azure.mgmt.cognitiveservices", reason="Install the optional foundry extra for SDK checks")

from azure.core.credentials import AccessToken
from azure.core.pipeline.transport import HttpResponse, HttpTransport
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient

from hf_cloud.core.deployment import DeploymentStatus
from hf_cloud.core.exceptions import DeploymentError, ProviderError
from hf_cloud.providers.foundry import FoundryProvider

SUB = "00000000-0000-0000-0000-000000000000"
BASE = f"/subscriptions/{SUB}/resourceGroups/rg/providers/Microsoft.CognitiveServices/accounts/account"
CONFIG = {
    "model_asset_id": "azureml://registries/azure-huggingface/models/example/versions/1",
    "deployment_template": "azureml://registries/azure-huggingface/deploymenttemplates/example-h100/versions/1",
    "accelerator_type": "H100_80GB",
}


class Response(HttpResponse):
    def __init__(self, request, status, data):
        super().__init__(request, None)
        self.status_code = status
        self.headers = {"content-type": "application/json"}
        self.content_type = "application/json"
        self._data = json.dumps(data).encode() if data is not None else b""

    def body(self):
        return self._data

    def text(self, encoding=None):
        return self._data.decode(encoding or "utf-8")

    def json(self):
        return json.loads(self._data)

    def read(self):
        return self._data

    def iter_bytes(self, **kwargs):
        yield self._data

    def iter_raw(self, **kwargs):
        yield self._data

    def close(self):
        pass


class ArmTransport(HttpTransport):
    def __init__(self, state="Succeeded"):
        self.requests = []
        self.resource = None
        self.state = state

    def open(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def send(self, request, **kwargs):
        assert request.url.startswith("https://management.azure.com" + BASE)
        assert "api-version=2026-03-15-preview" in request.url
        self.requests.append(request)
        path = request.url.split("?", 1)[0]
        if path.endswith("/accounts/account"):
            return Response(
                request,
                200,
                {
                    "id": BASE,
                    "name": "account",
                    "kind": "AIServices",
                    "location": "eastus",
                    "properties": {"customSubDomainName": "account"},
                },
            )
        if request.method == "PUT":
            body = json.loads(request.body)
            assert "tags" not in body
            self.resource = {"id": BASE + "/managedComputeDeployments/test", "name": "test", **body}
            self.resource["properties"].update(
                {
                    "provisioningState": self.state,
                    "acceleratorsPerInstance": 1,
                    "provisioningDetails": {"message": "InsufficientComputeQuota"}
                    if self.state == "Failed"
                    else None,
                }
            )
            return Response(request, 200, self.resource)
        if request.method == "PATCH":
            body = json.loads(request.body)
            assert body == {"sku": {"name": "GlobalManagedCompute", "capacity": 2}}
            self.resource["sku"] = body["sku"]
            return Response(request, 200, self.resource)
        if request.method == "DELETE":
            self.resource = None
            return Response(request, 204, None)
        if path.endswith("/managedComputeDeployments"):
            return Response(request, 200, {"value": [self.resource] if self.resource else []})
        if self.resource:
            return Response(request, 200, self.resource)
        return Response(request, 404, {"error": {"code": "ResourceNotFound", "message": "Not found"}})


def provider_with_transport(monkeypatch, tmp_path, state="Succeeded"):
    monkeypatch.setattr("hf_cloud.core.config.Config.DEFAULT_CONFIG_FILE", tmp_path / "config.json")

    class Credential:
        def get_token(self, *args, **kwargs):
            return AccessToken("in-memory-test-token", int(time.time()) + 3600)

    credential = Credential()
    transport = ArmTransport(state)
    provider = FoundryProvider(SUB, "rg", "account")
    provider.client._credential = credential
    provider.client._management = CognitiveServicesManagementClient(credential, SUB, transport=transport)
    return provider, transport


def test_real_sdk_create_deserialize_list_scale_delete(monkeypatch, tmp_path):
    provider, transport = provider_with_transport(monkeypatch, tmp_path)
    deployment = provider.deploy("org/model", "test", CONFIG)
    assert deployment.status == DeploymentStatus.RUNNING
    assert deployment.deployment_id == BASE + "/managedComputeDeployments/test"
    assert deployment.instance_type == "H100_80GB"
    assert provider.list_deployments()[0].status == DeploymentStatus.RUNNING
    assert provider.update_deployment("test", {"instance_count": 2}).instance_count == 2
    assert provider.delete_deployment("test")
    assert [r.method for r in transport.requests].count("PUT") == 1
    assert [r.method for r in transport.requests].count("PATCH") == 1
    assert transport.resource is None


def test_real_sdk_failure_never_becomes_a_success_record(monkeypatch, tmp_path):
    provider, _ = provider_with_transport(monkeypatch, tmp_path, "Failed")
    with pytest.raises((DeploymentError, ProviderError), match="InsufficientComputeQuota"):
        provider.deploy("org/model", "test", CONFIG)
