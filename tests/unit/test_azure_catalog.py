"""Published model discovery, backend isolation and publisher-owned templates."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
import pytest

from hf_cloud.core.exceptions import ConfigurationError, ProviderError
from hf_cloud.providers.azure_catalog import AzureCatalog, PublishedModel, lookup_foundry_model

HTTP_CLIENT = httpx.Client
TEMPLATE = "azureml://registries/azure-huggingface/deploymenttemplates/qwen-h100/labels/latest"


class AzureError(Exception):
    def __init__(self, code):
        self.status_code = code
        super().__init__(str(code))


def model(name="org-model", version="7", **kwargs):
    return NS(
        name=name,
        version=version,
        tags={"task": "text-generation"},
        properties={},
        default_deployment_template=None,
        allowed_deployment_templates=[],
        **kwargs,
    )


def test_azureml_resolves_documented_name_and_pins_latest_once():
    catalog = AzureCatalog()
    client = Mock()
    catalog._clients["HuggingFace"] = client
    client.models.get.return_value = model()
    first = catalog.resolve("Org/Model", "azureml")
    assert first.asset_id == "azureml://registries/HuggingFace/models/org-model/versions/7"
    assert first.task == "text-generation"
    assert catalog.resolve("Org/Model", "azureml") is first
    assert catalog.resolve("Org/Model", "azureml", "7") is first
    client.models.get.assert_called_once_with(name="org-model", label="latest")


@pytest.mark.parametrize(
    "status,error", [(404, ConfigurationError), (403, ProviderError), (429, ProviderError)]
)
def test_registry_failures_are_clear_and_never_switch_backend(status, error):
    catalog = AzureCatalog()
    client = Mock()
    catalog._clients["HuggingFace"] = client
    client.models.get.side_effect = AzureError(status)
    with pytest.raises(error, match="catalog"):
        catalog.resolve("org/model", "azureml")
    assert set(catalog._clients) == {"HuggingFace"}


def install_lookup(monkeypatch, handler):
    client = HTTP_CLIENT(transport=httpx.MockTransport(handler), follow_redirects=False)
    monkeypatch.setattr(httpx, "Client", Mock(return_value=client))


def test_lookup_uses_documented_service_and_distinguishes_classic(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "url": "https://ai.azure.com/nextgen/goto/discover/models/org--model/version/3/registry/azure-huggingface"
            },
        )

    install_lookup(monkeypatch, handler)
    assert lookup_foundry_model("org/model") == ("azure-huggingface", "org--model", "3")
    assert requests[0].url.params["model_id"] == "org/model"
    assert "authorization" not in requests[0].headers
    install_lookup(
        monkeypatch,
        lambda _: httpx.Response(
            200, json={"url": "https://ai.azure.com/explore/models/org-model/version/1/registry/HuggingFace"}
        ),
    )
    with pytest.raises(ConfigurationError, match="azureml deploy"):
        lookup_foundry_model("org/model")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"url": "https://evil.test/explore/models/org-model/version/1/registry/azure-huggingface"},
        {"url": "https://ai.azure.com@evil.test/explore/models/org-model/version/1/registry/azure-huggingface"},
        {"url": "https://ai.azure.com/explore/models/org-model/version/1/registry/azure-huggingface?token=foo"},
    ],
)
def test_lookup_rejects_unexpected_urls(monkeypatch, payload):
    install_lookup(monkeypatch, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(ProviderError, match="resolve"):
        lookup_foundry_model("org/model")


@pytest.mark.parametrize(
    "status,expected", [(404, ConfigurationError), (500, ProviderError), (302, ProviderError)]
)
def test_lookup_failures_never_fabricate_an_asset(monkeypatch, status, expected):
    install_lookup(monkeypatch, lambda _: httpx.Response(status, headers={"location": "https://evil.test"}))
    with pytest.raises(expected):
        lookup_foundry_model("org/model")


def test_foundry_pins_model_and_default_template_and_accelerator(monkeypatch):
    monkeypatch.setattr(
        "hf_cloud.providers.azure_catalog.lookup_foundry_model",
        lambda _: ("azure-huggingface", "org--model", "3"),
    )
    catalog = AzureCatalog()
    client = Mock()
    catalog._clients["azure-huggingface"] = client
    published = model("org--model", "3")
    published.default_deployment_template = NS(asset_id=TEMPLATE)
    client.models.get.return_value = published
    client.deployment_templates.get.return_value = NS(
        name="qwen-h100",
        version="5",
        stage="Active",
        accelerator_maps=[
            NS(accelerator_type="H100_80GB", number_of_accelerators_per_model_instance=2, default=True),
        ],
    )
    result = catalog.foundry_template(catalog.resolve("org/model", "foundry"))
    assert result == {
        "model_asset_id": "azureml://registries/azure-huggingface/models/org--model/versions/3",
        "deployment_template": "azureml://registries/azure-huggingface/deploymenttemplates/qwen-h100/versions/5",
        "accelerator_type": "H100_80GB",
        "accelerators_per_instance": 2,
    }
    client.deployment_templates.get.assert_called_once_with(name="qwen-h100", label="latest")


def test_ambiguous_or_incompatible_templates_are_not_guessed():
    catalog = AzureCatalog()
    published = PublishedModel(
        "org/model",
        "azure-huggingface",
        "org--model",
        "3",
        templates=(TEMPLATE, TEMPLATE.replace("h100", "a100")),
    )
    with pytest.raises(ConfigurationError, match="unambiguous"):
        catalog.foundry_template(published)
    with pytest.raises(ConfigurationError, match="not published"):
        catalog.foundry_template(published, TEMPLATE.replace("qwen", "other"))
    client = Mock()
    catalog._clients["azure-huggingface"] = client
    client.deployment_templates.get.return_value = NS(
        name="qwen-h100",
        version="5",
        stage="Active",
        accelerator_maps=[
            NS(accelerator_type="H100_80GB", number_of_accelerators_per_model_instance=1, default=False),
            NS(accelerator_type="A100_80GB", number_of_accelerators_per_model_instance=1, default=False),
        ],
    )
    with pytest.raises(ConfigurationError, match="accelerator-type"):
        catalog.foundry_template(published, "qwen-h100")
    assert catalog.foundry_template(published, "qwen-h100", "A100_80GB")["accelerator_type"] == "A100_80GB"
