# HF-Cloud

A CLI tool for managing HuggingFace model deployments across multiple cloud providers.

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/hf-cloud.git
cd hf-cloud

# Install base CLI
pip install -e .

# Install with SageMaker support
pip install -e ".[sagemaker]"

# Install with all providers
pip install -e ".[all]"

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Configure your provider (Optional)

You can configure some default settings for your cloud provider. This step is optional, as you can also provide these settings via command-line arguments during deployment or use the default configuration from your environment.

```bash
hf-cloud providers configure sagemaker
```

### 2. Deploy a model

```bash
hf-cloud sagemaker deploy gpt2 \
  --name my-gpt2-endpoint \
  --instance-type ml.g5.xlarge \
  --region us-east-1
```

### 3. Check deployment status

```bash
hf-cloud sagemaker status my-gpt2-endpoint
```

### 4. Test inference

```bash
hf-cloud sagemaker invoke my-gpt2-endpoint \
  --input "Once upon a time"
```

### 5. List all deployments

```bash
hf-cloud ls
```

### 6. Delete deployment

```bash
hf-cloud sagemaker delete my-gpt2-endpoint
```

## Commands

| Command | Description |
|---------|-------------|
| `hf-cloud [PROVIDER] deploy <model>` | Deploy a model to the specified provider |
| `hf-cloud [PROVIDER] ls` | List deployments for the specified provider |
| `hf-cloud [PROVIDER] describe <id>` | Show deployment details |
| `hf-cloud [PROVIDER] status <id>` | Check deployment status |
| `hf-cloud [PROVIDER] logs <id>` | View deployment logs |
| `hf-cloud [PROVIDER] invoke <id>` | Test inference |
| `hf-cloud [PROVIDER] delete <id>` | Delete deployment |
| `hf-cloud ls` | List all deployments (all providers) |
| `hf-cloud providers ls` | List available providers |
| `hf-cloud providers configure <provider>` | Configure provider credentials |

## Supported Providers

- **AWS SageMaker** - Fully implemented
- **Azure ML** - Work in progress
- **Google Cloud Vertex AI** - Work in progress

## Requirements

- Python 3.12+

You will also need to have the respective cloud provider authentication set up, so that `hf-cloud` can access your account. Current provider-specific requirements:

- **AWS SageMaker**: AWS credentials configured via AWS CLI or environment variables.
- **Azure ML**: Azure CLI logged in or service principal set up.
- **Google Cloud Vertex AI**: Google Cloud SDK authenticated or service account key set up.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
```

## License

Apache-2.0
