# HF-Cloud

A CLI tool for managing HuggingFace model deployments across multiple cloud providers.

## Installation modes

HF-Cloud can be used in two equivalent ways:

- Standalone CLI install: commands are prefixed with `hf-cloud ...`
- Hugging Face Hub CLI extension install: commands are prefixed with `hf cloud ...`

Preferred method: install and run it as an HF CLI extension (`hf cloud ...`). More details about HF CLI extensions [here](https://huggingface.co/docs/huggingface_hub/guides/cli-extensions).

In the examples below, replace `<cli>` with either `hf-cloud` or `hf cloud`.

Both installation modes include the AWS SageMaker and Google Cloud Vertex AI
dependencies by default. Azure ML is not yet implemented.

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

# Install the CLI with SageMaker and Vertex AI support
pip install -e .

# Install with development dependencies
uv sync --group dev
```

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
<cli> sagemaker estimate meta-llama/Llama-2-7b-hf
<cli> vertex estimate meta-llama/Llama-2-7b-hf
```

Both estimate commands accept `--safety-margin` as the fraction of VRAM to
reserve, from `0` (inclusive) to `1` (exclusive). For example,
`--safety-margin 0.2` keeps 20% free. Without this option, the provider's bundled
instance catalog supplies the default. Use `--json` for machine-readable results
or `--all` to show all compatible instances.

Estimation uses hf-mem v1's GPU-fit assessment for each GPU layout, including
per-device overhead and KV-cache replication. The safety margin is passed as
`--gpu-memory-utilization = 1 - safety-margin`; it is applied once. Instances
are checked in ascending aggregate GPU memory order, with catalog order breaking
ties. Instances with the same GPU count and memory per GPU reuse an assessment.
`--all` checks every distinct layout and can take longer than a minimum estimate.

Use a namespaced Hub ID such as `openai-community/gpt2` (bare `gpt2` is not
accepted by hf-mem v1), a Hub URL, or a local model directory:

```bash
<cli> sagemaker estimate openai-community/gpt2 \
  --max-model-len 4096 --batch-size 4 --kv-cache-dtype fp8

<cli> vertex estimate ./local-model --max-model-len 8192 --json

# Select a GGUF quantization instead of estimating an ambiguous file listing
<cli> vertex estimate org/model-GGUF --format gguf --file Q4_K_M
```

Both commands also accept `--revision` and `--max-num-batched-tokens` (use `2048`
for A100). Context length defaults to the model config, batch size to one, and KV
cache dtype to `auto`. Authentication uses `HF_TOKEN` or the saved Hub login.
GGUF and ONNX repositories with multiple weight files require `--file`; available
files are shown in the error. PyTorch pickle-only weights are unsupported.

Memory is displayed in **GiB**. Existing JSON keys ending in `Gb` are retained
and also represent GiB, with `memoryUnit` identifying the unit. Each match includes
its own `requiredVramGb`, `neededGpuCount`, and `maxModelLen` (the longest context
that fits, or zero when not applicable). With `--all`, the top-level
`requiredVramGb` describes the first match. JSON includes the hf-mem version,
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
dependencies. It runs in a separate Python 3.10+ environment, so HF-Cloud's own
Python requirement is unchanged.

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
| `<cli> [PROVIDER] estimate <model>` | Estimate minimum viable instance for a model |
| `<cli> [PROVIDER] delete <id>` | Delete deployment |
| `<cli> ls` | List all deployments (all providers) |
| `<cli> providers ls` | List available providers |
| `<cli> providers configure <provider>` | Configure provider credentials |
| `<cli> skills preview` | Preview generated SKILL.md for AI assistants |
| `<cli> skills add` | Install SKILL.md and optionally symlink to assistant folders |

## Supported Providers

- **AWS SageMaker** - Fully implemented
- **Google Cloud Vertex AI** - Fully implemented
- **Azure ML** - Work in progress

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

# Estimate minimum viable instance
<cli> sagemaker estimate meta-llama/Llama-2-7b-hf
# Show all compatible instances
<cli> sagemaker estimate meta-llama/Llama-2-7b-hf --all

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

# Estimate minimum viable machine/GPU configuration
<cli> vertex estimate meta-llama/Llama-2-7b-hf
# JSON output for scripting
<cli> vertex estimate meta-llama/Llama-2-7b-hf --json

# Delete
<cli> vertex delete my-gpt2-endpoint
```

## Requirements

- Python 3.9+

You will also need to have the respective cloud provider authentication set up, so that `hf-cloud` can access your account. Current provider-specific requirements:

- **AWS SageMaker**: AWS credentials configured via AWS CLI or environment variables.
- **Azure ML**: Azure CLI logged in or service principal set up.
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
