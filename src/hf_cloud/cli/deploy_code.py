"""Standalone SDK examples rendered from the same inputs as deploy commands."""

from pprint import pformat
from typing import Any, Optional

from hf_cloud.providers.sagemaker.utils import (
    get_sagemaker_env_vars,
    get_schema_samples,
    get_serving_settings,
    sanitize_endpoint_name,
)
from hf_cloud.providers.vertex.utils import get_serving_container_uri
from hf_cloud.providers.vertex.utils import sanitize_endpoint_name as vertex_endpoint_name


def _literal(value: Any) -> str:
    return pformat(value, width=100, sort_dicts=False)


def sagemaker_code(
    model_id: str,
    name: str,
    instance_type: str,
    count: int,
    region: str,
    role: Optional[str],
    task: str,
    tensor_parallel_size: Optional[int] = None,
) -> str:
    samples = get_schema_samples(task, model_id)
    settings = get_serving_settings(instance_type, tensor_parallel_size)
    env = get_sagemaker_env_vars(model_id, task=task, tensor_parallel_size=settings["tensor_parallel_size"])
    server = "    model_server=ModelServer.VLLM,\n" if task == "text-generation" else ""
    return f"""# Requires: sagemaker>=3.29.0, boto3, huggingface-hub
# Running this script creates billable cloud resources.
import json
import os
import boto3
from huggingface_hub import get_token
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.serve import ModelBuilder, ModelServer
from sagemaker.serve.builder.schema_builder import SchemaBuilder

session = boto3.Session(region_name={region!r})
sagemaker_session = Session(boto_session=session)
role = {role!r}
if role is None:
    role = os.environ.get("SAGEMAKER_ROLE_ARN") or None
if role is None:
    try:
        role = get_execution_role(sagemaker_session=sagemaker_session)
    except ValueError:
        role = "sagemaker_execution_role"
if not role.startswith("arn:"):
    role = session.client("iam").get_role(RoleName=role)["Role"]["Arn"]
# The execution role must trust sagemaker.amazonaws.com and allow model access.
token = os.environ.get("HF_TOKEN") or get_token()
env = {_literal(env)}
if token:
    env["HF_TOKEN"] = token

schema = SchemaBuilder(**{_literal(samples)})
builder = ModelBuilder(
    model={model_id!r}, role_arn=role, sagemaker_session=sagemaker_session,
    instance_type={instance_type!r}, schema_builder=schema, env_vars=env,
    model_metadata={{"HF_TASK": env["HF_TASK"]}},
{server})
builder.build()
predictor = builder.deploy(
    endpoint_name={sanitize_endpoint_name(name)!r}, initial_instance_count={count!r},
    instance_type={instance_type!r}, container_timeout_in_seconds={settings["container_timeout_in_seconds"]!r},
    wait=True,
)
endpoint = session.client("sagemaker").describe_endpoint(EndpointName={sanitize_endpoint_name(name)!r})
if endpoint["EndpointStatus"] != "InService":
    raise RuntimeError(endpoint.get("FailureReason") or endpoint["EndpointStatus"])
print(endpoint["EndpointArn"])

# Send a sample request once the endpoint is ready.
res = predictor.invoke(
    body=json.dumps({_literal(samples["sample_input"])}),
    content_type="application/json",
)
print(json.loads(res.body.read()))
"""


def vertex_code(model_id: str, name: str, config: dict[str, Any], task: str) -> str:
    endpoint = vertex_endpoint_name(name)
    image = get_serving_container_uri(
        location=config["location"],
        machine_type=config["machine_type"],
        accelerator_type=config["accelerator_type"],
    )
    return f"""# Requires: google-cloud-aiplatform, huggingface-hub
# Running this script creates billable cloud resources.
# Verify that the serving container supports this model's architecture and task.
import os
from google.cloud import aiplatform
from huggingface_hub import get_token

aiplatform.init(project={config["project"]!r}, location={config["location"]!r})
token = os.environ.get("HF_TOKEN") or get_token()
env = {{"HF_MODEL_ID": {model_id!r}, "HF_TASK": {task!r}}}
if token:
    env["HF_TOKEN"] = token
model = aiplatform.Model.upload(
    display_name={f"hf-{endpoint}"!r},
    serving_container_image_uri={image!r},
    serving_container_environment_variables=env,
    serving_container_predict_route="/predict", serving_container_health_route="/health",
    labels={{"hf-cloud": "true"}},
)
endpoint = aiplatform.Endpoint.create(display_name={endpoint!r}, labels={{"hf-cloud": "true"}})
model.deploy(
    endpoint=endpoint,
    deployed_model_display_name={f"hf-{model_id.lower().replace('/', '--')[:50]}"!r},
    machine_type={config["machine_type"]!r},
    accelerator_type={config["accelerator_type"]!r},
    accelerator_count={config["accelerator_count"] if config["accelerator_type"] else 0!r},
    min_replica_count={config["min_replica_count"]!r}, max_replica_count={config["max_replica_count"]!r},
)
print(endpoint.resource_name)
"""


def azureml_code(plan: dict[str, Any], client: Any) -> str:
    return f"""# Requires: azure-ai-ml>=1.35.2,<2, azure-identity
# Authenticate with az login or configure DefaultAzureCredential.
# Running this script creates billable cloud resources.
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, IdentityConfiguration
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

client = MLClient(DefaultAzureCredential(), {client.subscription_id!r}, {client.resource_group!r}, {client.workspace_name!r})
endpoint = ManagedOnlineEndpoint(
    **{_literal(plan["endpoint"])},
    identity=IdentityConfiguration(type="system_assigned"),
)
deployment = ManagedOnlineDeployment(**{_literal(plan["deployment"])})
try:
    client.online_endpoints.get(name=endpoint.name)
except ResourceNotFoundError:
    pass
else:
    raise RuntimeError("Endpoint already exists; choose a new name")
client.online_endpoints.begin_create_or_update(endpoint).result()
client.online_deployments.begin_create_or_update(deployment).result()
deployment = client.online_deployments.get(name=deployment.name, endpoint_name=endpoint.name)
if deployment.provisioning_state != "Succeeded":
    raise RuntimeError(f"Deployment did not succeed: {{deployment.provisioning_state}}")
endpoint = client.online_endpoints.get(name=endpoint.name)
endpoint.traffic = {_literal(plan["traffic"])}
client.online_endpoints.begin_create_or_update(endpoint).result()
print(endpoint.scoring_uri)
"""


def foundry_code(resource: dict[str, Any], scope: dict[str, str], client: Any) -> str:
    return f"""# Requires: azure-mgmt-cognitiveservices==15.0.0b2, azure-identity
# Authenticate with az login or configure DefaultAzureCredential.
# Running this script creates billable cloud resources.
from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.core.exceptions import ResourceNotFoundError

client = CognitiveServicesManagementClient(DefaultAzureCredential(), {client.subscription_id!r})
scope = {_literal(scope)}
resource = {_literal(resource)}
try:
    client.managed_compute_deployments.get(**scope)
except ResourceNotFoundError:
    pass
else:
    raise RuntimeError("Deployment already exists; choose a new name")
client.managed_compute_deployments.begin_create_or_update(**scope, resource=resource).result()
deployment = client.managed_compute_deployments.get(**scope)
if deployment.properties.provisioning_state != "Succeeded":
    raise RuntimeError(f"Deployment did not succeed: {{deployment.properties.provisioning_state}}")
print(deployment.id)
"""
