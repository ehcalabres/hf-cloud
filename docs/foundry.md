# Microsoft Foundry managed compute

This backend uses current Foundry managed compute: published Hugging Face models,
compatible deployment templates and dedicated accelerators. Microsoft supplies the
runtime, image and security patching. HF Cloud does not maintain containers or
publish arbitrary Hub weights. See the [Microsoft guide](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/hugging-face-models)
and [Hugging Face guide](https://huggingface.co/docs/microsoft-azure/guides/managed-compute-foundry).

For an Azure ML workspace or Foundry classic hub-based project, use the
[Azure ML catalog backend](azureml.md). The two catalogs and quota pools differ;
HF Cloud requires an explicit backend selection.

## Setup and first deployment

```bash
uv sync --extra foundry
az login
uv run hf-cloud providers configure foundry
uv run hf-cloud foundry catalog Qwen/Qwen3-32B
uv run hf-cloud foundry deploy Qwen/Qwen3-32B --name qwen-test --dry-run
```

Configure an existing Foundry **AIServices account**, with `subscription_id`,
`resource_group` and `account_name`. Its name is the Foundry resource/account name,
rather than the project name. Configure the optional HTTPS base `endpoint` only
when needed. Environment variables `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`,
`AZURE_FOUNDRY_ACCOUNT` and `AZURE_FOUNDRY_ENDPOINT` override saved settings.
Explicit scope options override these and go **before the subcommand**:

```bash
hf-cloud foundry --subscription-id SUBSCRIPTION --resource-group RG --account ACCOUNT ls
```

The optional extra pins the preview management SDK
`azure-mgmt-cognitiveservices==15.0.0b2` and includes Azure Identity, Monitor and
HTTP clients. Use the equivalent `pip install 'hf-cloud[foundry]'` for pip installs.
Deployments use API `2026-03-15-preview`; model/template discovery uses the Azure ML
SDK already included in HF Cloud. Authenticate with `DefaultAzureCredential`, for
example through `az login`.

Your identity needs account deployment-management permissions, such as Cognitive
Services Contributor. Inference needs Foundry User. You also need available
**Foundry managed compute quota**, which is separate from Azure ML VM quota.
This service is in public preview; check its availability and terms before use.
HF Cloud does not create accounts or assign roles.

## Hub IDs, models and templates

To compare the two Microsoft catalogs in one command, use
`hf-cloud catalog MODEL_ID` (or add `--json`). It checks Azure ML and Foundry
independently, distinguishes missing models from checks that failed, and links to
available model pages. No workspace/account defaults are needed for discovery.


`foundry catalog MODEL_ID` uses the
[documented Hub-to-Azure lookup](https://huggingface.co/docs/microsoft-azure/foundry/models),
then verifies the published model in the `azure-huggingface` registry. It lists
that model version's allowed template references and reports the publisher's
default selection, when unambiguous. If the lookup points to the classic
`HuggingFace` registry, the command directs you to `azureml` instead.

`foundry deploy MODEL_ID --name NAME` resolves the Hub ID and uses the published
default template and accelerator family. With only one allowed choice, that choice
can be used. HF Cloud never guesses between multiple choices with no default or
claims the publisher's default is the cheapest. To choose explicitly:

```bash
hf-cloud foundry deploy Qwen/Qwen3-32B --name qwen-test \
  --template TEMPLATE_NAME_FROM_CATALOG --accelerator-type H100_80GB --dry-run
```

The template must be published for the selected model version. Its accelerator
maps determine supported families and GPUs per model instance. `--instance-count`
controls model replicas, defaulting to one; it does not change GPUs inside a
replica. `--catalog-version` selects a specific published model version.
Automatically resolved model and template references are pinned to concrete
versions. The default `NoAutoUpgrade` policy preserves the selected model version.
Change `--version-upgrade-option` explicitly if you want Microsoft's upgrade policy.

A dry run reads the public lookup and authenticated registry metadata and prints
the exact resource request. It creates no resources or local deployment state.
It cannot verify available project quota or reserve capacity. Remove `--dry-run`
to provision. The command waits, re-reads the result and only reports success for
`Succeeded` with the requested accelerator and replica count. Failures exit nonzero;
inspect and remove any resources left by Azure.

## Advanced profiles and explicit references

Reusable profiles remain available when you want to pin model/template inputs,
share known template metadata, or preview without catalog access:

```bash
hf-cloud foundry deploy Qwen/Qwen3-32B --name qwen-test \
  --profile examples/foundry/qwen3-32b-h100.json --dry-run
```

With a complete explicit profile, dry run is offline. Review sample asset versions
and availability in your project before provisioning; samples are not automatically
refreshed. Profiles may also include context and price data for estimates below.
Alternatively, provide the complete `--model-asset-id`, `--template` asset URI and
`--accelerator-type`. Those references must be compatible published assets.

Published catalog weights are required. Arbitrary private Hub models and Hub
tokens are not supported by this deployment path. For published gated models,
use Azure ML / Foundry classic as described in the [gated model guide](azureml.md#published-gated-models).

## Verify, invoke and clean up

```bash
hf-cloud foundry ls
hf-cloud foundry describe qwen-test
hf-cloud foundry status qwen-test
hf-cloud foundry invoke qwen-test --input 'What is GPU memory?'
hf-cloud foundry update qwen-test --instance-count 2
hf-cloud foundry logs qwen-test --tail 50
hf-cloud foundry delete qwen-test
```

Chat invocation uses Microsoft Entra authentication through the account's
`https://ACCOUNT.services.ai.azure.com/openai/v1/` endpoint. The OpenAI request's
`model` field is the **deployment name**. For embeddings and other OpenAI routes,
use the appropriate `--route` and `--json-file`:

```bash
hf-cloud foundry invoke EMBEDDINGS_DEPLOYMENT --route embeddings --json-file request.json
```

Non-OpenAI routes use the dedicated managed-deployment route. Route inputs must be
relative paths; credentials are never sent to arbitrary URLs or redirects.
JSON invocation does not support streaming. Full deployment ARM IDs are checked
against the selected account, and local state isolates identical names across
accounts. `hf-cloud ls --refresh` visits the configured account and accounts
already recorded in local state.

Scaling preserves model/template properties and updates replica count. Deletion
removes only the selected managed-compute deployment, leaving the account and
project intact. Logs currently report recent Azure Activity Log management events,
not GPU process stdout. Runtime monitoring depends on the
[Foundry/Azure Monitor surface](https://learn.microsoft.com/en-us/azure/foundry/how-to/diagnostic-logging).

## Compare template estimates

Foundry estimates work automatically, without a profile:

```bash
hf-cloud estimate Qwen/Qwen3-14B
hf-cloud foundry estimate Qwen/Qwen3-14B --max-model-len 4096 --hours 24 --instance-count 2
```

HF Cloud reads allowed templates for the published model version, resolves template
labels to concrete versions, and discovers each accelerator family, GPU count and
machine-readable context limit. Template aliases are deduplicated. Archived or
unreadable templates and templates without known GPU memory/context metadata are
skipped; `--verbose` explains why. Other providers can still be compared if Foundry
discovery fails or exceeds the shared 15-second catalog deadline.

Foundry managed compute is billed per accelerator-hour, using the accelerator
count in the selected template. HF Cloud bundles a shared USD rate table for
`GlobalManagedCompute`: A100 80 GB $3.95, H100 80 GB $7.91, H200 141 GB $9.10 and
MI300X 192 GB $7.91 per accelerator-hour. These are a 2026-11-13 snapshot of
[Microsoft's public pricing table](https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/microsoft/),
supplied by the user because the automated page reader returned placeholders.
They are bundled estimates, not live quotes. A price entry
does not establish model/hardware availability: only published templates are
considered. Families without a known rate remain unpriced and cannot win cheapest.

Foundry prices cannot be inferred from Azure ML VM prices, even for the same GPU
family: they are separate services and billing units. Automatic deployment hints
pin the estimated model version, template version and accelerator family.

Profiles remain optional overrides for explicit template selection or a verified
rate, including account-specific pricing. Keep metadata in a profile, including `acceleratorsPerInstance`,
`maxModelLen`, `pricingSource`, `pricingObservedAt` and `pricingRegion`, with **one**
of these rate fields:

- `pricePerAcceleratorHourUsd`: the verified hourly rate for one accelerator.
  HF Cloud multiplies it by `acceleratorsPerInstance` to obtain the instance rate.
- `pricePerInstanceHourUsd`: the complete hourly rate for one model instance,
  already including all of its accelerators.

The two fields are mutually exclusive. Runtime duration and replica count then
apply to the instance rate. This follows [Microsoft's billing formula](https://learn.microsoft.com/en-us/azure/foundry/concepts/managed-compute-overview#billing-quota-and-deployment-scopes).
Use current rates from the pricing calculator for your accelerator family and
scope when overriding the bundled rates. HF Cloud does not fetch a live Foundry
quote or substitute Azure VM rates.

```bash
hf-cloud foundry estimate Qwen/Qwen3-32B \
  --profile examples/foundry/qwen3-32b-h100.json --max-model-len 4096 --hours 24 --instance-count 2
hf-cloud estimate Qwen/Qwen3-32B \
  --foundry-profile examples/foundry/qwen3-32b-h100.json --max-model-len 4096
```

The second command compares the supplied template with SageMaker, Vertex and Azure
ML hardware catalogs in one hf-mem metadata session. Microsoft model publication is
verified first. For Foundry, the profile must name this Hub model's published
version and an active allowed template; its accelerator family and GPU count must
match the published template. Model asset references must use a concrete version.
Template labels are resolved and checked against allowed concrete versions.
Stale/incompatible profiles and failed checks are excluded before price ranking.
Root estimates automatically rank published Foundry templates with usable bundled
rates; profiles replace automatic template selection and pricing when supplied.
A template's declared context
cap must support the requested workload. The Foundry template itself controls
runtime/context settings; the estimate does not rewrite it. Unknown prices sort
last and cannot win cheapest. The launch price in the example profile is historical;
verify your current rate before using it as a quote. Automatic estimates join
published templates with the bundled Foundry rate table, never with VM prices.

Cost output leads with the comparison, followed by a compact provider availability
table. SageMaker/Vertex runtime support is unverified; Microsoft publication is
checked separately from memory fit and prices. Add
`--verbose` to see full publication/profile errors and price sources. JSON includes
the complete check results and projected costs regardless of verbosity. `--hours`
is a cost scenario, not a shutdown schedule; clean up deployments when finished.

## Verification

Catalog discovery, ambiguous defaults, offline profiles, SDK resource serialization,
failed provisioning, scaling and inference routing are covered by offline tests.
The public model lookup has also been checked against current catalog responses.
No live Foundry resources were provisioned while implementing this change.
