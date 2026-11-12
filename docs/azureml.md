# Azure ML managed online endpoints

Azure ML is HF Cloud's primary Microsoft backend, alongside SageMaker and Vertex
AI. It loads a Hugging Face Hub model ID directly in a vLLM container on a managed
online endpoint. Choose the VM size and number of replicas; GPUs within a VM use
tensor parallelism. There is no Foundry catalog lookup, registry model ID or
published template requirement. [Foundry](foundry.md) remains a separate advanced
preview for Microsoft's published catalog/template experience.

## Setup

```bash
uv sync
az login
uv run hf-cloud providers configure azureml
```

The regular installation includes `azure-ai-ml>=1.35.2,<2` and
`azure-identity>=1.27.0,<2`. Authentication uses `DefaultAzureCredential`, including
Azure CLI login, service principals and managed identity. You need an existing
Azure ML workspace, its associated storage/container registry, GPU quota and
permission to manage online endpoints, deployments and environments. This command
does not create a workspace or assign roles on your behalf.

Configuration records `subscription_id`, `resource_group`, `workspace_name` and
`default_instance_type`. Environment variables `AZURE_SUBSCRIPTION_ID`,
`AZURE_RESOURCE_GROUP`, `AZUREML_WORKSPACE_NAME` override saved scope defaults.
Explicit group flags override both and go **before the subcommand**:

```bash
hf-cloud azureml --subscription-id SUBSCRIPTION --resource-group RG --workspace WORKSPACE ls
```

Use the same scope for subsequent commands. Full endpoint ARM IDs are validated
against the selected workspace. Local state stores full ARM IDs, so identical
endpoint names in different workspaces do not collide. `hf-cloud ls --refresh`
refreshes the default workspace and every workspace already present in local state.

## Estimate and deploy

```bash
hf-cloud estimate Qwen/Qwen3-0.6B --max-model-len 2048
hf-cloud estimate Qwen/Qwen3-0.6B --provider azureml --max-model-len 2048
hf-cloud azureml estimate Qwen/Qwen3-0.6B --max-model-len 2048
hf-cloud azureml deploy Qwen/Qwen3-0.6B --name qwen-test \
  --instance-type Standard_NC24ads_A100_v4 --instance-count 1 \
  --max-model-len 2048 --dry-run
```

The dry run validates the local plan and prints the environment, model arguments,
secret references and endpoint request. It makes no SDK, authentication or cloud
calls and writes no deployment state. It cannot verify model/image compatibility,
permissions, regional capacity or quotas. Remove `--dry-run` to provision.

The default is one `Standard_NC24ads_A100_v4` VM and a serving component named
`main`. `--instance-count` scales replicas; `--deployment-name` after `deploy`
changes the component name. Endpoint/component names must be 3–32 lowercase
letters, digits or hyphens, start with a letter and end with a letter or digit.
An existing endpoint is never overwritten by `deploy`.

Before creating resources, HF Cloud checks the selected VM's GPU count using the
workspace region's size metadata. Sizes outside the bundled catalog need explicit
`--gpu-count`; it must match Azure's metadata. This is a layout check, not a capacity
reservation or a quota check. Azure may still reject a SKU for online deployment.

Azure ML builds a small environment in the workspace registry from the pinned
`vllm/vllm-openai:v0.32.0` image and an argument-based Python entrypoint. You don't
need local Docker. Environment versions depend on the image, entrypoint and
scoring route, so matching environments are reused across models. The container
loads weights from the Hub at startup; no local weight download or Azure model
registration is required. Workspace build infrastructure and outbound Hub access
must be configured, including in network-restricted workspaces.

`--image` accepts another tagged or digest-pinned image containing the vLLM Python
server. Use an image compatible with the GPU's installed driver and your model's
architecture. This backend currently targets NVIDIA GPUs and vLLM-supported
models; a memory fit alone does not establish runtime compatibility.
`--revision` pins the Hub revision; `--trust-remote-code` explicitly allows model
repository code to execute. Public models need no token.

Workload options include `--max-model-len`, `--batch-size` (maximum active
sequences), `--kv-cache-dtype`, `--max-num-batched-tokens`, `--dtype` and
`--safety-margin`. The memory margin defaults to 0.1 and maps to vLLM's 0.9 GPU
memory utilization. T4 catalog instances default to half precision. Use the same
workload settings for estimates and deployment; unspecified context length can
cause the server to request the model's full context and exceed available memory.

## Private and gated models

Accept the model's Hub license terms with the account owning your token. Create
an Azure ML **CustomKeys workspace connection** containing a secret credential
named `HF_TOKEN`, then pass its connection name:

```bash
hf-cloud azureml deploy org/private-model --name private-test \
  --hf-token-connection hub-token --max-model-len 2048 --dry-run
```

The plan contains `${{azureml://connections/hub-token/credentials/HF_TOKEN}}`,
which Azure resolves into the container environment. HF Cloud requests connection
metadata with `populate_secrets=False`; it does not read or save the secret value.
Your local `HF_TOKEN` or `hf auth login` session is not forwarded automatically.

This uses Azure's [secret injection preview](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoint-with-secret-injection?view=azureml-api-2).
HF Cloud creates a system-assigned endpoint identity and sets
`enforce_access_to_default_secret_stores=enabled`. The deploying identity needs
workspace connection secret-reader permission for Azure to grant the endpoint's
identity access automatically. Ask your Azure administrator to configure the
required permissions before deployment; missing access or a missing credential
causes Azure provisioning to fail. See Microsoft's guide for CustomKeys connection
creation and the `Azure Machine Learning Workspace Connection Secrets Reader` role.

## Invoke, inspect, scale and delete

```bash
hf-cloud azureml status qwen-test
hf-cloud azureml describe qwen-test
hf-cloud azureml logs qwen-test --tail 100
hf-cloud azureml logs qwen-test --container storage-initializer
hf-cloud azureml invoke qwen-test --input 'Explain GPU memory in one sentence.'
hf-cloud azureml update qwen-test --instance-count 2
hf-cloud azureml delete qwen-test
```

Endpoints use Microsoft Entra token authentication (`aad_token`). Invokers need
endpoint scoring permission; managing an endpoint and invoking it are separate
permissions. The SDK obtains the token and sends the request to the scoring URI.
`--task text-generation` (default) selects `/v1/completions` and `--input` becomes
`prompt`. `--task chat-completion` selects `/v1/chat/completions` and creates a user
message. `--task feature-extraction` selects `/v1/embeddings`, enables vLLM's pooling
runner and supplies `input`. Use `--json-file request.json` for richer requests on
the configured route. HF Cloud supplies the original Hub ID as the served model
name. JSON invocation does not support streaming.

`ls` lists HF Cloud-tagged endpoints. `describe`, `status`, `logs`, `update` and
`invoke` also accept an existing single-deployment endpoint by name or ARM ID.
For endpoints created outside HF Cloud, invocation requires a vLLM-compatible
request in `--json-file` with its actual served `model` name. Select an ambiguous component with `azureml --deployment-name BLUE ...` before
the subcommand. Scaling changes only that component's replica count and preserves
its environment, model and traffic.

Deployment reports success only after both endpoint and serving component are
`Succeeded`, with the requested instance layout and 100% traffic to that component.
Failed provisioning exits nonzero and is not saved as a successful local deployment.
Azure can leave resources after failure: inspect `status`, `logs` and Azure ML's
activity logs, then delete the endpoint if it is no longer needed. Missing serving
components remain visible as incomplete endpoints. Creation and scaling are blocking
SDK operations; large downloads/builds can take substantial time.

Deletion removes the **endpoint and all components under it**, releasing their
compute. It leaves the workspace, shared environments, registry images and secret
connections intact. These assets can incur charges independently.

## Pricing and validation

The bundled Azure ML catalog records T4, A100 and H100 GPU layouts, with an
`eastus` Linux Consumption USD/VM-hour snapshot from **2026-11-12** obtained through
the [Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices).
It excludes Spot, Low Priority and Windows rates. Estimates compare GPU memory fit
and one VM-hour, not throughput or full deployment cost. Multiply by replicas and
add storage, network, environment builds and registry costs. Azure regional rates,
SKU availability and your negotiated prices can differ. Catalog hardware-source
links are retained in `src/hf_cloud/data/instances/azureml.json`.

The implementation is checked with offline endpoint lifecycle tests, real SDK
entity serialization, and the shared estimator tests. No live Azure endpoint has
been provisioned as part of this change. See Microsoft's [managed endpoint overview](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online?view=azureml-api-2),
[custom container guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-container?view=azureml-api-2)
and [vLLM Docker documentation](https://docs.vllm.ai/en/latest/deployment/docker/).
