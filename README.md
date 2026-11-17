# HF-Cloud

A CLI tool for managing HuggingFace model deployments across multiple cloud providers.

## Installation modes

HF-Cloud can be used in two equivalent ways:

- Standalone CLI install: commands are prefixed with `hf-cloud ...`
- Hugging Face Hub CLI extension install: commands are prefixed with `hf cloud ...`

Preferred method: install and run it as an HF CLI extension (`hf cloud ...`). More details about HF CLI extensions [here](https://huggingface.co/docs/huggingface_hub/guides/cli-extensions).

In the examples below, replace `<cli>` with either `hf-cloud` or `hf cloud`.

Both installation modes include AWS SageMaker, Google Cloud Vertex AI and
Azure ML support by default. Microsoft Foundry uses the optional `foundry` extra.

## Hugging Face CLI extension

This project is packaged as a Python `hf` CLI extension. It can be discovered and installed using the following commands:

```bash
# Search for available extensions
hf extensions search

# Install this extension from GitHub
hf extensions install ehcalabres/hf-cloud

# Run extension commands
hf cloud --help
hf cloud sagemaker ls
```

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/ehcalabres/hf-cloud.git
cd hf-cloud

# Install the CLI with SageMaker, Vertex AI and Azure ML support
pip install -e .

# Install with development dependencies
uv sync --group dev
```

### Provider SDK versions

Deployments use the SDKs installed in the environment running HF Cloud. Dependency
minimums are declared in `pyproject.toml`; `uv sync` uses the versions in `uv.lock`.
It does not upgrade packages just because a newer release exists. To upgrade one
SDK explicitly, use `uv lock --upgrade-package sagemaker` followed by `uv sync`.
SageMaker uses its SDK's `ModelBuilder` to build the serving image. Vertex AI uses
the serving image configured in HF Cloud. SDK updates do not update running endpoints.

During `sagemaker deploy`, `vertex deploy` and `azureml deploy`, HF Cloud checks
for newer stable, non-yanked SDK releases on PyPI in the background and prints an informational notice
when the command finishes. SageMaker checks `sagemaker` and `boto3`; Vertex checks
`google-cloud-aiplatform` and `google-cloud-logging`; Azure ML checks `azure-ai-ml`
and `azure-identity`. Results, including failed checks, are cached for 24 hours in `~/.hf-cloud/cache/sdk-versions/`. Checks never
delay deployment for a network response or install updates. If the command finishes
before the check, a later invocation can use the cached result. Offline failures
are silent during deployment; estimates do not run SDK checks.

Inspect installed and latest versions without cloud credentials:

```bash
hf-cloud providers versions
hf-cloud providers versions --provider sagemaker --refresh
hf-cloud providers versions --json
```

The report identifies releases that require a newer Python version. Other dependency
constraints are left to your package manager; an available release is not a guarantee
of compatibility. Set `HF_CLOUD_NO_UPDATE_CHECK=1` to disable automatic checks.
Explicit `providers versions` commands still check PyPI.

## Microsoft deployments: published catalog models

HF Cloud deploys published models using Microsoft/Hugging Face serving assets.
It does not maintain custom Microsoft serving images or import arbitrary Hub
repositories. Both paths accept a Hugging Face model ID:

| Command | Azure resource | Use when |
|---|---|---|
| `azureml` | Azure ML workspace or Foundry classic hub-based project | You want a managed online endpoint with a VM size and replica count |
| `foundry` | Current Foundry account (`AIServices`) | You want Foundry managed compute with a published template and accelerator family (preview) |

### Find where a model is available

```bash
uv run hf-cloud catalog deepseek-ai/DeepSeek-V4.1-Flash
uv run hf-cloud catalog deepseek-ai/DeepSeek-V4.1-Flash --json
```

This checks Azure ML and current Foundry independently and links to each available
catalog model. It distinguishes **Available**, **Not available**, and **Could not
verify** (login, network or service failures). Registry reads use Azure credentials;
run `az login` first. No workspace/account configuration is required for discovery.
Checks run concurrently with a 15-second overall deadline and are fresh on each
command. An incomplete `catalog` check exits nonzero, while a confirmed absence
is a successful discovery result. No cloud resources are created.

### Azure ML / Foundry classic

```bash
uv sync
az login
uv run hf-cloud providers configure azureml
uv run hf-cloud azureml catalog Qwen/Qwen2.5-32B-Instruct
uv run hf-cloud azureml deploy Qwen/Qwen2.5-32B-Instruct --name qwen-test \
  --instance-type Standard_NC40ads_H100_v5 --dry-run --export-dir ./azure-deployment
```

Catalog lookup verifies and pins the published model version. Remove `--dry-run`
to provision, or use the exported definitions with Azure CLI and your production
pipeline. The catalog supplies the runtime and scoring configuration. See the
[Azure ML guide](docs/azureml.md) for workspace setup, gated models, export,
invocation and cleanup.

### Current Microsoft Foundry

```bash
uv sync --extra foundry
az login
uv run hf-cloud providers configure foundry
uv run hf-cloud foundry catalog Qwen/Qwen3-32B
uv run hf-cloud foundry deploy Qwen/Qwen3-32B --name qwen3-32b --dry-run
```

HF Cloud resolves the published model, template and accelerator defaults. If the
publisher provides multiple choices without a default, select a template shown by
`foundry catalog` using `--template NAME`. Review the plan, then remove `--dry-run`
to provision. Explicit asset references and reusable profiles remain available.
See the [Foundry guide](docs/foundry.md) for templates, authentication, quotas,
pricing comparisons and cleanup. This deployment type is in preview.

Azure ML and current Foundry use different registries and quotas. A model may be
published in one catalog and unavailable in the other; HF Cloud reports that
before creating resources and never switches providers automatically.

## Agent Skills

HF-Cloud can scaffold a reusable `SKILL.md` for coding agents.

```bash
# Preview generated skill content
<cli> skills preview

# Install skill in project-level central directory (.agents/skills/hf-cloud)
<cli> skills add

# Install to a custom destination directory
<cli> skills add --dest ./my-skills

# Link into specific assistant directories
<cli> skills add --codex
<cli> skills add --claude --cursor

# Overwrite existing skill files/symlinks
<cli> skills add --force
```

## Quick Start

### 1. Configure your provider (Optional)

You can configure some default settings for your cloud provider. This step is optional, as you can also provide these settings via command-line arguments during deployment or use the default configuration from your environment.

```bash
<cli> providers configure sagemaker
```

### 2. Deploy a model

```bash
<cli> sagemaker deploy gpt2 \
  --name my-gpt2-endpoint \
  --instance-type ml.g5.xlarge \
  --region us-east-1
```

Use `--role` to override the configured SageMaker execution role for one deployment:

```bash
<cli> sagemaker deploy openai-community/gpt2 --name my-endpoint \
  --instance-type ml.g6.2xlarge --region us-east-1 \
  --role arn:aws:iam::123456789012:role/MySageMakerExecutionRole
```

Or resolve the role by name in the current AWS account:

```bash
<cli> sagemaker deploy openai-community/gpt2 --name my-endpoint \
  --instance-type ml.g6.2xlarge --region us-east-1 \
  --role-name MySageMakerExecutionRole
```

Use either `--role` or `--role-name`; both override the configured default. Before
building the model, HF Cloud reads the role with IAM `GetRole` and checks its trust
policy for `sts:AssumeRole` by `sagemaker.amazonaws.com`. Missing roles, unreadable
policies, explicit denies, and incompatible source account/region restrictions fail
before creating deployment resources. The caller needs `iam:GetRole` permission.
Conditions needing deployment context are reported and deferred to AWS. This
preflight checks trust; AWS still evaluates `iam:PassRole`, organization policies,
and runtime permissions during deployment. See [SageMaker execution roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

SageMaker deployment waits for the endpoint and verifies its AWS status. Only an
`InService` endpoint is reported and saved as `running`. Failed endpoints produce
a nonzero exit code and show AWS's `FailureReason`, when available. A failure to
verify status is also an error; resources are retained for inspection.

### 3. Check deployment status

```bash
<cli> sagemaker status my-gpt2-endpoint
```

### 4. Test inference

```bash
<cli> sagemaker invoke my-gpt2-endpoint \
  --input "Once upon a time"
```

### 5. List all deployments

```bash
<cli> ls
```

### 6. Estimate required instance for a model

```bash
# Compare all providers by catalog hourly price
<cli> estimate deepseek-ai/DeepSeek-V4.1-Flash

# Show the most expensive compatible instances first
<cli> estimate deepseek-ai/DeepSeek-V4.1-Flash --sort most-expensive --limit 10

# Filter providers, configure the workload, or get structured results
<cli> estimate openai-community/gpt2 --provider sagemaker --provider vertex \
  --max-model-len 512 --batch-size 4 --json

# Compare instances within one provider
<cli> estimate meta-llama/Llama-2-7b-hf --provider sagemaker
<cli> estimate meta-llama/Llama-2-7b-hf --provider vertex
```

Project continuous compute costs with a duration and replica count:

```bash
<cli> estimate Qwen/Qwen2.5-32B-Instruct --provider azureml --hours 24 --instance-count 2
<cli> foundry estimate Qwen/Qwen3-14B --hours 24 --instance-count 2
```

The total is the hourly **instance rate × replicas × running hours**. GPU counts
inside an instance are already included in its rate. The default remains a
one-instance hourly estimate. `--hours` only changes the estimate; it does not
schedule endpoint shutdown. Storage, networking, autoscaling and other service
charges are outside this projection. Deploy hints use the projected replica count.

The cost comparison appears first, followed by a compact **Provider availability**
table for every selected provider. Deployment evidence and cost availability are
separate: SageMaker and Vertex show **Runtime unverified** because they deploy from
the Hub without requiring a catalog entry, while Azure ML and Foundry show their
publication check results. Memory fit alone does not establish runtime support,
Hub permissions, regional capacity or quota. Selections the current deploy commands
cannot reproduce (such as specific Hub revisions, files or local weights) are
marked **Selection unsupported**, even when their memory/cost estimates are useful.
The table respects `--provider` and summarizes every match regardless of `--limit`.
Use `--verbose` on `estimate` or `catalog` for full check explanations;
on estimates it also shows individual price sources. JSON always retains the
complete check results and pricing metadata, plus `costEstimate`,
`deploymentPricePerHourUsd` and `estimatedCostUsd`.

The primary `<cli> estimate` command compares SageMaker, Vertex AI and Azure ML
instance catalogs, plus published Foundry templates with bundled accelerator rates. It highlights the cheapest
compatible instance and lists matches cheapest-first by default.
`--sort most-expensive` reverses the price order, and `--limit` limits displayed
rows while the cheapest summary still considers every compatible match. Unknown or
invalid prices are excluded from the cheapest summary and sort after priced instances.
The comparison shares one model-metadata session across all providers.
Human-readable output ends with a deployment command for the cheapest compatible
instance, including Vertex AI's accelerator type and count. The hint considers all
matches regardless of `--sort` or `--limit`; change `my-endpoint` to your endpoint
name. Hints are omitted for local models, selected files/revisions and unsupported
weight formats. Microsoft model publication is checked before matching or ranking:
unavailable or unverified providers are excluded, with reasons shown in the output.
Azure ML hints pin the verified catalog version. The published model controls the
runtime and context settings; configure serving workloads on providers that support
runtime options. Publication checks do not guarantee SKU compatibility or quota.

These are **bundled catalog estimates in USD per instance-hour**, not live
quotes. Catalog dates and pricing regions (or their absence) are displayed and
included in JSON. Rates can differ by region, serving service, and discounts;
storage, network, and other charges are outside this comparison. Verify a
deployment's price with [AWS pricing](https://aws.amazon.com/sagemaker/ai/pricing/),
[Google Cloud pricing](https://cloud.google.com/vertex-ai/pricing), or
[Azure pricing](https://azure.microsoft.com/en-us/pricing/details/machine-learning/).

The Vertex AI snapshot was checked on **2026-11-17** for **Iowa (`us-central1`)**,
using the public USD on-demand **Prediction and explanation** tables.
Rates cover one full node, including
its CPUs, host RAM and all GPUs; do not multiply them by GPU count again. G2, A2,
A3 Mega and A3 Ultra use published node-hour rates. A3 High estimates sum Google's
published A3 prediction CPU/RAM/H100 component rates and the
[documented machine specifications](https://cloud.google.com/vertex-ai/docs/predictions/configure-compute);
the inputs are recorded in each entry's `pricingCalculation` in the catalog.
These rates exclude storage, networking, Spot/reservation pricing and commitment
discounts. Other regions can differ; this snapshot does not check regional capacity.

Cloud capacity and quotas are not checked. Foundry template layouts and context
limits are discovered from the published catalog automatically; per-accelerator
rates are multiplied by the template's GPU count. The bundled Global Managed
Compute USD rates are an independently verified 2026-11-13 snapshot of Microsoft's
pricing page, not live quotes.
Families without known rates remain unpriced. Templates without readable context
limits or GPU layouts are skipped with details under `--verbose`.
Use `--foundry-profile PROFILE.json` as an optional override for a verified rate
or explicit template selection. Azure ML VM rates are never reused for Foundry.
See [Foundry estimates](docs/foundry.md#compare-template-estimates) for template
metadata and pricing inputs.

Global JSON output includes `cheapest`, `mostExpensive`, `totalCompatible`,
`pricedCount`, `unpricedCount`, ranked `instances` with their `provider`, and
`pricing` metadata. Estimates also include `providerAvailability` with each
provider's deployment status/path, cost-estimate status and explanation. This
summary reuses existing checks without additional Hub or cloud requests.
Microsoft results also include `catalogAvailability` and
`excludedProviders` when applicable. Unknown checks are excluded from recommendations,
while other providers can still produce a successful estimate. `--provider` can be repeated to select providers. A shared
`--safety-margin` defaults to 0.1 so the comparison uses the same reserve policy
for every provider. The model-selection and workload options below also apply.

The older provider spellings remain as shortcuts:

```bash
# These produce the same results and accept the same sorting/workload options
<cli> estimate MODEL --provider sagemaker
<cli> sagemaker estimate MODEL
```

`<cli> vertex estimate MODEL` is likewise an alias for `estimate MODEL --provider
vertex`. The defaults are identical: list all compatible instances in cheapest-first
order with a 10% safety margin. `--all` / `-a` remains accepted for older scripts;
all matches are listed by default. Use `--limit` to shorten the output.

Provider aliases now return the same JSON schema as the top-level command.
Scripts using the older singular `instance` key should use `cheapest` or the
ranked `instances` array instead. Provider aliases no longer select by minimum
VRAM or take their default safety margin from the provider catalog.

All estimate commands accept `--safety-margin` as the fraction of VRAM to
reserve, from `0` (inclusive) to `1` (exclusive). For example,
`--safety-margin 0.2` keeps 20% free. The default is 0.1 for every provider.
Use `--json` for machine-readable results.

Estimation uses hf-mem v1's GPU-fit assessment for each GPU layout, including
per-device overhead and KV-cache replication. The safety margin is passed as
`--gpu-memory-utilization = 1 - safety-margin`; it is applied once. GPU layouts
are assessed in ascending aggregate GPU memory order. All layouts run in one
isolated worker, reusing the repository listing,
configuration, weight metadata, and parsed Safetensors summaries within that
command. Instances with the same GPU count and memory per GPU also reuse an
assessment. Every command assesses all catalog layouts to rank compatible
instances by price; `--limit` only limits the displayed rows.

Use a namespaced Hub ID such as `openai-community/gpt2` (bare `gpt2` is not
accepted by hf-mem v1), a Hub URL, or a local model directory:

```bash
<cli> estimate openai-community/gpt2 --provider sagemaker \
  --max-model-len 512 --batch-size 4 --kv-cache-dtype fp8

<cli> estimate ./local-model --provider vertex --max-model-len 8192 --json

# Select a GGUF quantization instead of estimating an ambiguous file listing
<cli> estimate org/model-GGUF --provider vertex --format gguf --file Q4_K_M
```

All estimate commands also accept `--revision` and `--max-num-batched-tokens` (use `2048`
for A100). Context length defaults to the model config, batch size to one, and KV
cache dtype to `auto`. Authentication uses `HF_TOKEN` or the saved Hub login.
GGUF and ONNX repositories with multiple weight files require `--file`; available
files are shown in the error. PyTorch pickle-only weights are unsupported.

Memory is displayed in **GiB**. Existing JSON keys ending in `Gb` are retained
and also represent GiB, with `memoryUnit` identifying the unit. Each match includes
its own `requiredVramGb`, `neededGpuCount`, and `maxModelLen` (the longest context
that fits, or zero when not applicable). The `cheapest` and `mostExpensive`
entries use the same match format. JSON includes the hf-mem version,
workload options, actual estimated context length, warnings, and CPU-offloaded
weight memory (`cpuMemoryGb` and exact `cpuMemoryBytes`). Host RAM must be checked
separately from GPU fit.

These estimates follow hf-mem's serving-engine assumptions; configure the actual
deployment's engine, sharding, context length and concurrency to match. Estimating
local weights, GGUF or ONNX does not add deployment support for those formats.

Until v1 is released, HF-Cloud runs the reviewed hf-mem commit
[`2448e7c`](https://github.com/alvarobartt/hf-mem/commit/2448e7c6457949951495b6dba2f01ac3bebdf36a)
through `uvx`. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and Git; the first estimate needs network access to install hf-mem and its
dependencies. It runs in a separate Python 3.10+ environment to isolate its
dependencies from HF-Cloud's provider SDKs.

### 7. Delete deployment

```bash
<cli> sagemaker delete my-gpt2-endpoint
```

## Commands

| Command | Description |
|---------|-------------|
| `<cli> [PROVIDER] deploy <model>` | Deploy a model to the specified provider |
| `<cli> [PROVIDER] ls` | List deployments for the specified provider |
| `<cli> [PROVIDER] describe <id>` | Show deployment details |
| `<cli> [PROVIDER] status <id>` | Check deployment status |
| `<cli> [PROVIDER] logs <id>` | View deployment logs |
| `<cli> [PROVIDER] invoke <id>` | Test inference |
| `<cli> [PROVIDER] estimate <model>` | Compatibility shortcut for `estimate <model> --provider PROVIDER` |
| `<cli> estimate <model>` | Compare compatible instances across providers by hourly price |
| `<cli> [PROVIDER] delete <id>` | Delete deployment |
| `<cli> ls` | List all deployments (all providers) |
| `<cli> providers ls` | List available providers |
| `<cli> providers configure <provider>` | Configure provider credentials |
| `<cli> providers versions` | Inspect installed provider SDKs and stable PyPI releases |
| `<cli> catalog <model>` | Show availability in Azure ML and Foundry side by side |
| `<cli> azureml catalog <model>` | Check and pin a published Azure ML model |
| `<cli> foundry catalog <model>` | Resolve a published Foundry model and its templates |
| `<cli> azureml update <id> --instance-count N` | Scale Azure ML endpoint replicas |
| `<cli> foundry update <id> --instance-count N` | Scale Foundry model instances |
| `<cli> skills preview` | Preview generated SKILL.md for AI assistants |
| `<cli> skills add` | Install SKILL.md and optionally symlink to assistant folders |

## Supported Providers

- **AWS SageMaker** - Fully implemented
- **Google Cloud Vertex AI** - Fully implemented
- **Microsoft Azure ML** - Published catalog models on managed online endpoints (also Foundry classic)
- **Microsoft Foundry** - Published Hub models and template discovery on managed compute (preview)

## Provider-Specific Examples

### AWS SageMaker

Every SageMaker endpoint command accepts `--region` (`-r`). Region selection uses
the command option, then the HF-Cloud `default_region` setting, then the AWS
session's region (environment or selected AWS profile), and finally `us-east-1`.
Use the deployment's region when checking, invoking, or deleting its endpoint:

```bash
<cli> sagemaker status my-gpt2-endpoint --region eu-west-1
<cli> sagemaker ls --region eu-west-1
<cli> sagemaker delete my-gpt2-endpoint --region eu-west-1
```

Local state distinguishes deployments by provider, region, and endpoint ID.
`<cli> ls --refresh` refreshes SageMaker endpoints in the default region and all
SageMaker regions recorded in local state.

Refresh saves each provider or region's results in a single batch. Local state
and configuration files are replaced atomically to avoid partial JSON files if
a write fails. Simultaneous CLI processes are not coordinated by a file lock.

```bash
# Configure (optional)
<cli> providers configure sagemaker

# Deploy
<cli> sagemaker deploy gpt2 \
  --name my-gpt2-endpoint \
  --instance-type ml.g5.xlarge \
  --region us-east-1

# Check status
<cli> sagemaker status my-gpt2-endpoint

# Invoke
<cli> sagemaker invoke my-gpt2-endpoint --input "Hello world"

# Compare compatible instances within SageMaker
<cli> estimate meta-llama/Llama-2-7b-hf --provider sagemaker
# Show the five cheapest matches
<cli> estimate meta-llama/Llama-2-7b-hf --provider sagemaker --limit 5

# Delete
<cli> sagemaker delete my-gpt2-endpoint
```

### Google Cloud Vertex AI

```bash
# Configure (optional)
<cli> providers configure vertex

# Deploy
<cli> vertex deploy gpt2 \
  --name my-gpt2-endpoint \
  --machine-type n1-standard-4 \
  --location us-central1

# Deploy with GPU
<cli> vertex deploy meta-llama/Llama-2-7b-hf \
  --name my-llama-endpoint \
  --machine-type n1-standard-8 \
  --accelerator-type NVIDIA_TESLA_T4 \
  --accelerator-count 1

# Check status
<cli> vertex status my-gpt2-endpoint

# Invoke
<cli> vertex invoke my-gpt2-endpoint --input "Hello world"

# Compare compatible Vertex AI configurations
<cli> estimate meta-llama/Llama-2-7b-hf --provider vertex
# JSON output for scripting
<cli> estimate meta-llama/Llama-2-7b-hf --provider vertex --json

# Delete
<cli> vertex delete my-gpt2-endpoint
```

## Requirements

- Python 3.10+ (required by SageMaker SDK 3.29.0)

You will also need to have the respective cloud provider authentication set up, so that `hf-cloud` can access your account. Current provider-specific requirements:

- **AWS SageMaker**: AWS credentials configured via AWS CLI or environment variables.
- **Microsoft Foundry**: Foundry optional extra installed; `az login`, managed identity, or service principal credentials; a configured AIServices account and approved managed-compute GPU quota.
- **Google Cloud Vertex AI**: Google Cloud SDK authenticated or service account key set up.

## Development

Development dependencies require Python 3.10 or newer; `.python-version` selects Python 3.12.

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run linting
uv run ruff check src tests

# Opt-in integration tests against the pinned real hf-mem CLI using tiny local
# synthetic weights (first run needs dependency network access; no cloud resources)
HF_CLOUD_TEST_HF_MEM=1 uv run pytest tests/integration/test_hf_mem_v1.py
```
