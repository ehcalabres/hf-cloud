# Microsoft Foundry managed compute

HF Cloud implements the current **Foundry managed compute** deployment path described
in [Microsoft's Hugging Face guide](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/hugging-face-models)
and the [HF managed-compute guide](https://huggingface.co/docs/microsoft-azure/guides/managed-compute-foundry).
This uses `Microsoft.CognitiveServices/accounts/managedComputeDeployments` and
`GlobalManagedCompute`. It does not create Azure ML managed online endpoints.

Microsoft manages the runtime and GPU topology. You select a published registry
model, a compatible deployment template, an accelerator family, and model instance
count. The template determines GPUs per instance, runtime, and context settings.
See [Microsoft's architecture and template overview](https://learn.microsoft.com/en-us/azure/foundry/concepts/managed-compute-overview).

## Setup

```bash
uv sync --extra foundry
az login
uv run hf-cloud providers configure foundry
```

For pip installations use `pip install 'hf-cloud[foundry]'`. The optional extra pins
`azure-mgmt-cognitiveservices==15.0.0b2`, the preview SDK documented by Microsoft;
its managed-compute operations use API `2026-03-15-preview`. Azure Identity and
Azure Monitor clients are included. Other providers don't require these packages.

Use an existing Foundry account of kind `AIServices`, rather than an Azure ML
workspace or an OpenAI-only account. Configure its subscription, resource group,
account name, and optionally its HTTPS account base endpoint. `DefaultAzureCredential`
supports Azure CLI login, managed identities, and service principal environment
credentials. Credentials and Hub tokens are never stored in the deployment state.

You can also set `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`,
`AZURE_FOUNDRY_ACCOUNT`, and optionally `AZURE_FOUNDRY_ENDPOINT`.
Explicit scope options go **before** the subcommand:

```bash
hf-cloud foundry --subscription-id SUBSCRIPTION_ID --resource-group RESOURCE_GROUP \
  --account ACCOUNT_NAME status DEPLOYMENT_NAME
```

Creating/scaling/deleting requires the control-plane role described in Microsoft's
guide (for example Cognitive Services Contributor). Inference uses Foundry User
and the account-level Entra token scope. Managed-compute GPU quota is separate
from VM quota. This feature is currently a public preview.

## Select published assets

In the Foundry model catalog, filter by **Hugging Face**, **Managed compute**, and
**Available in my Project**. Copy the complete model asset ID and compatible
deployment-template ID from the model card/deployment wizard, plus the template's
accelerator family. HF Cloud accepts A100_80GB, H100_80GB, and MI300_192GB.

```bash
hf-cloud foundry deploy Qwen/Qwen3-32B --name qwen3-32b \
  --model-asset-id azureml://registries/azure-huggingface/models/qwen--qwen3-32b/versions/3 \
  --template azureml://registries/azure-huggingface/deploymenttemplates/qwen--qwen3-32b--40k-nvidia-h100/labels/latest \
  --accelerator-type H100_80GB --instance-count 1 --dry-run
```

`--dry-run` prints the SDK request without Azure authentication or resource changes.
Remove it to deploy. The Hub ID records the intended upstream model; HF Cloud
doesn't guess a registry slug/version from it or automatically publish a Hub repo.
Alternatively, pass the complete registry model asset URI as the positional model.
Azure verifies that the selected model, template, and accelerator are compatible.

To reuse your team's published references, put them in a JSON profile:

```bash
hf-cloud foundry deploy Qwen/Qwen3-32B --name qwen3-32b \
  --profile examples/foundry/qwen3-32b-h100.json --dry-run
```

A profile requires `modelId`, `modelAssetId`, `deploymentTemplateId`, and
`acceleratorType`. It is an HF Cloud wrapper around published asset references,
not an Azure template publishing format. The example IDs are from the HF guide;
select the current compatible references in your project before provisioning.
Both `/versions/...` and `/labels/...` registry references are supported. Prefer
versioned model/template references for reproducibility; labels can move.

The default model version policy is `NoAutoUpgrade`. Use `--version-upgrade-option`
to opt into `OnceNewDefaultVersionAvailable` or `OnceCurrentVersionExpired`.
Runtime patching remains managed by Microsoft.

## Deploy, inspect, invoke, scale, delete

```bash
hf-cloud foundry deploy Qwen/Qwen3-32B --name qwen3-32b \
  --profile examples/foundry/qwen3-32b-h100.json
hf-cloud foundry ls
hf-cloud foundry describe qwen3-32b
hf-cloud foundry status qwen3-32b
hf-cloud foundry invoke qwen3-32b --input "What is the capital of France?"
hf-cloud foundry update qwen3-32b --instance-count 2
hf-cloud foundry logs qwen3-32b --tail 50
hf-cloud foundry delete qwen3-32b --yes
hf-cloud ls --refresh --provider foundry
hf-cloud providers versions --provider foundry
```

Deployment and scaling wait for the operation and re-read provisioning status.
Only `Succeeded` is saved as `running`; failed operations return a nonzero exit
code with Azure's error or provisioning details. Existing deployment names are
not overwritten by `deploy`. Scaling patches capacity while preserving the model
and template. Deletion waits for the managed deployment to be removed; it never
deletes the account, project, or registry assets. Failed resources are retained
for inspection and explicit cleanup.

Names and full ARM deployment IDs are accepted. Local records use full ARM IDs so
the same deployment name in different accounts/subscriptions doesn't collide.
Global refresh visits the configured account and accounts in local records.

Inference uses the account's dedicated managed-deployment route and Entra ID,
without retrieving API keys. Chat uses the deployment name in the `model` field.
For other model routes, provide the path from the model's documentation and a JSON
request. For example, an embeddings runtime can use:

```bash
hf-cloud foundry invoke EMBEDDINGS_DEPLOYMENT --route embeddings --json-file request.json
```

Requests use HTTPS, never follow redirects, and don't support streaming in this
JSON command. Unsupported chat routes can return 404; select the model's actual
route rather than assuming every model supports chat completions.

`logs` retrieves **Azure Monitor management activity over the last 24 hours** for
the selected deployment, with monitoring read permissions. It doesn't claim to
expose runtime container stdout. Request diagnostics and runtime metrics use the
Foundry/Azure Monitor surface described in [Microsoft's monitoring guidance](https://learn.microsoft.com/en-us/azure/foundry/how-to/diagnostic-logging).

## Compare template estimates

Foundry doesn't sell arbitrary VM sizes for this deployment path. A useful
comparison needs a model-specific template rather than a generic GPU catalog.
Add `acceleratorsPerInstance` and `maxModelLen` to the profile. Set the context cap
to the published template limit or a lower conservative limit. A template whose
context cap is below the requested/inferred context is excluded from matches.

```bash
hf-cloud foundry estimate Qwen/Qwen3-32B \
  --profile examples/foundry/qwen3-32b-h100.json --max-model-len 4096
hf-cloud estimate Qwen/Qwen3-32B \
  --foundry-profile examples/foundry/qwen3-32b-h100.json --max-model-len 4096
```

The second command compares SageMaker, Vertex, and the supplied Foundry template
in the same hf-mem metadata session. Root estimates without a Foundry profile
continue to use the bundled SageMaker/Vertex catalogs. The cheapest deployment
hint uses `foundry deploy ... --profile ...` when that template wins.

Supply `pricePerInstanceHourUsd`, `pricingSource`, `pricingObservedAt`, and
`pricingRegion` for price ranking. The rate is per **model instance**, including
all GPUs in that instance; it is not multiplied by GPU count again. Omit prices
that you haven't verified: unknown prices sort last and cannot win cheapest.
Foundry-only output without a usable price cannot declare a cheapest result.

The example contains the historical launch rate for one H100 from
[Microsoft's June 3, 2026 announcement](https://devblogs.microsoft.com/foundry/announcing-foundry-managed-compute/).
It is not a live quote. Update the profile for your pricing/region; quotas,
availability, runtime behavior, and model/template compatibility are checked by
Azure, not proven by hf-mem's memory assessment.

## Validation

Unit coverage exercises lifecycle errors, name/account scoping, profiles, price
ranking, context caps, Entra inference, and dry runs. Optional SDK tests run the
real management client against an in-memory ARM transport, covering request
serialization, resource deserialization, long-running operations, and failures:

```bash
uv run --extra foundry pytest tests/integration/test_foundry_sdk.py -q
```

A real deployment still needs an account, compatible published assets, and approved
managed-compute quota. These SDK tests create no Azure resources.
