# Azure ML / Foundry classic catalog deployments

This backend deploys **published Hugging Face catalog models** to Azure ML managed
online endpoints. It also works with a Foundry classic hub-based project by using
that project's name as the workspace. Microsoft/Hugging Face supply the runtime,
container and scoring configuration. HF Cloud does not build images, register
arbitrary model weights, or maintain serving scripts.

For the current Foundry portal's managed compute experience, use the separate
[Foundry backend](foundry.md). The two paths have different resources and quotas;
HF Cloud never switches between them automatically.

## Start with a published model

```bash
uv sync
az login
uv run hf-cloud providers configure azureml
uv run hf-cloud azureml catalog Qwen/Qwen2.5-32B-Instruct
uv run hf-cloud azureml deploy Qwen/Qwen2.5-32B-Instruct --name qwen-test \
  --instance-type Standard_NC40ads_H100_v5 --dry-run
```

`catalog` checks the `HuggingFace` Azure ML registry and reports the published
model name, concrete version, task (when published) and asset URI. HF IDs use the
[documented catalog name mapping](https://huggingface.co/docs/microsoft-azure/foundry/examples/deploy-large-language-models),
but HF Cloud verifies the asset exists rather than assuming any Hub model can be
deployed. Missing models or registry access fail before resources are created.
Try `hf-cloud foundry catalog MODEL_ID` if the model is available only in current
Foundry, or request an addition through the model's Hub Deploy menu.

`--dry-run` reads the catalog using Azure credentials and prints the pinned plan.
It creates no endpoint, deployment, image or local deployment state. It cannot
reserve capacity or verify quota and runtime startup. Remove `--dry-run` to deploy.
`--catalog-version VERSION` selects a specific published version; otherwise the
latest label is resolved once per command and the resulting concrete version is
used for deployment. A catalog version is independent of a Hub revision.

Select hardware recommended by the model's catalog page. The default VM size is
your configured `default_instance_type`, falling back to
`Standard_NC24ads_A100_v4`. HF Cloud checks regional size metadata before creating
an endpoint; this does not guarantee managed-endpoint capacity or model/SKU
compatibility. `--instance-count` controls replicas, defaulting to one. The serving
component is named `main`; `--deployment-name` after `deploy` changes it.

The published asset chooses its inference engine, GPU topology, context settings
and scoring schema. Flags for custom images, vLLM arguments, `--revision` and
`--trust-remote-code` are intentionally unavailable. A published deployment may
have a different runtime or context length than a memory estimate.

## Workspace and identity

Use an existing Azure ML workspace with its associated resources and sufficient
quota. Find it in [Azure ML Studio](https://ml.azure.com), or search for Azure
Machine Learning in [Azure Portal](https://portal.azure.com). A current Foundry
project is a different resource; a classic hub-based project can be used here.

Configuration stores `subscription_id`, `resource_group`, `workspace_name` and
`default_instance_type`. `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP` and
`AZUREML_WORKSPACE_NAME` override saved scope settings. Explicit group options take
precedence and go **before the subcommand**:

```bash
hf-cloud azureml --subscription-id SUBSCRIPTION --resource-group RG --workspace WORKSPACE ls
```

Authentication uses `DefaultAzureCredential`, including `az login`, service
principals and managed identity. Registry discovery needs catalog read access;
deployment also needs permission to manage online endpoints and deployments.
Endpoints use Microsoft Entra authentication (`aad_token`). Scoring permission is
separate from management permission. HF Cloud does not create workspaces or assign
roles. Full endpoint ARM IDs are checked against the chosen workspace; local state
keeps equal endpoint names in different workspaces separate.

## Export a starting point for production

```bash
hf-cloud azureml deploy Qwen/Qwen2.5-32B-Instruct --name qwen-test \
  --instance-type Standard_NC40ads_H100_v5 --catalog-version 1 \
  --dry-run --export-dir ./azure-deployment
```

This produces `endpoint.json` and `deployment.json`, including the concrete model
asset version and selected hardware. JSON is accepted by Azure CLI's YAML loader.
Exports never overwrite existing files. Review and version these definitions,
then add your organization's networking, scaling, monitoring and deployment policy.
They are a starting point, rather than a complete production infrastructure setup.
You can provision a fresh endpoint with the Azure CLI:

```bash
az ml online-endpoint create --file azure-deployment/endpoint.json -g RG -w WORKSPACE
az ml online-deployment create --file azure-deployment/deployment.json -g RG -w WORKSPACE --all-traffic
```

Exports contain secret references, never token values. No custom environment,
Dockerfile or scoring script is exported.

## Published gated models

Only models published in the Azure ML catalog are supported; arbitrary private Hub
repositories are not imported. For a published gated model, accept its Hub terms
and create a CustomKeys workspace connection containing `HF_TOKEN`. Then use:

```bash
hf-cloud azureml deploy org/published-gated-model --name gated-test \
  --hf-token-connection hub-token --dry-run
```

HF Cloud injects `${{azureml://connections/hub-token/credentials/HF_TOKEN}}` using
Azure's secret injection feature and enables default secret-store access on the
endpoint. It reads connection metadata without retrieving secrets. Your local Hub
login/token is not forwarded automatically. The deploying and endpoint identities
need the permissions described in the [gated model guide](https://huggingface.co/docs/microsoft-azure/guides/access-gated-models)
and [Azure secret injection guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoint-with-secret-injection?view=azureml-api-2).

## Operate and clean up

```bash
hf-cloud azureml status qwen-test
hf-cloud azureml describe qwen-test
hf-cloud azureml logs qwen-test --tail 100
hf-cloud azureml invoke qwen-test --input 'What is GPU memory?'
hf-cloud azureml update qwen-test --instance-count 2
hf-cloud azureml delete qwen-test
```

Invocation uses the model's published scoring route. `--input` supplies
`{"inputs": "..."}`; known text-generation tasks also receive
`parameters.max_new_tokens`. Use `--json-file` with the catalog's documented
request format for other tasks or richer requests. The request body is preserved;
HF Cloud does not force an OpenAI model field or replace the scoring server.

Success requires the endpoint and serving component to be `Succeeded`, with the
requested hardware, replicas and traffic. Failures exit nonzero without recording
a successful deployment. Azure may leave resources: inspect logs/activity logs
and delete unused endpoints. Incomplete endpoints remain visible. Scaling keeps
the existing published model and runtime. Deletion removes the endpoint and all
its components; the workspace and shared resources remain.

The root `ls --refresh` visits configured and previously recorded workspaces.
`azureml ls` shows HF Cloud-tagged endpoints. For an existing endpoint with multiple
components, select one using `azureml --deployment-name BLUE ...` before the
subcommand. Lifecycle operations remain available for previously created HF Cloud
endpoints, including those using the removed custom runtime.

## Estimates and verification

`hf-cloud catalog MODEL_ID` checks Azure ML and current Foundry side by side and
links to their published model pages. Discovery requires Azure login but no
workspace configuration. `--provider azureml` or `--provider foundry` narrows it;
`--json` provides machine-readable results. Failed/incomplete checks exit nonzero.

`hf-cloud estimate MODEL_ID --provider azureml` verifies publication before any
Azure ML row can be ranked or recommended. It excludes missing models, unknown
checks (authentication/network/service errors), and weight selections that catalog
deployments cannot follow, such as local files and Hub revisions. Other providers
can still produce estimates. Root estimates report Foundry availability too; a
verified template profile is required to include Foundry prices.

Checks run concurrently with a 15-second total deadline and do not persist negative
results between commands. Estimates expose `catalogAvailability` and
`excludedProviders` in JSON. Azure ML deploy hints include the verified
`--catalog-version`. Publication does not prove that the model's packaged runtime
matches the estimated workload, nor does it establish SKU compatibility, project
quota or regional capacity. Those still need deployment validation. Foundry uses
separate template pricing; Azure VM prices cannot be used as Foundry prices.

The Azure VM snapshot is `eastus`, Linux Consumption, observed 2026-11-12 through
the [Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices).
Add replicas, storage and networking to calculate deployment cost. SDK serialization,
export loading and endpoint lifecycle are tested offline; no live endpoint was
provisioned while implementing this change.
