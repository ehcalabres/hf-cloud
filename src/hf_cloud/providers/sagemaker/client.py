"""AWS SageMaker client wrapper."""

import os
import re
from typing import Any, Optional

from rich.console import Console
from rich.text import Text

from hf_cloud.core.config import Config
from hf_cloud.core.exceptions import AuthenticationError, ProviderError
from hf_cloud.providers.sagemaker.roles import validate_role_trust

console = Console()


class SageMakerClient:
    """Wrapper for AWS SageMaker and related clients."""

    def __init__(self, region: Optional[str] = None):
        """Initialize SageMaker client.

        Args:
            region: AWS region
        """
        self.region = self.resolve_region(region)
        self._sagemaker_client = None
        self._sagemaker_runtime_client = None
        self._logs_client = None
        self._iam_client = None
        self._session = None

    @staticmethod
    def resolve_region(region: Optional[str] = None) -> str:
        """Resolve explicit, HF-Cloud, AWS session, then fallback region."""
        if region:
            return region
        configured = Config().get_provider_config("sagemaker").get("default_region")
        if configured:
            return configured
        try:
            import boto3
        except ImportError:
            raise ProviderError(
                "sagemaker",
                "boto3 is not installed. Install with: pip install --upgrade hf-cloud",
            )
        return boto3.Session().region_name or "us-east-1"

    def _get_boto3_session(self) -> Any:
        """Get or create the session for this client's region."""
        if self._session is None:
            try:
                import boto3

                self._session = boto3.Session(region_name=self.region)
            except ImportError:
                raise ProviderError(
                    "sagemaker",
                    "boto3 is not installed. Install with: pip install --upgrade hf-cloud",
                )
        return self._session

    @property
    def sagemaker(self) -> Any:
        """Get SageMaker client."""
        if self._sagemaker_client is None:
            session = self._get_boto3_session()
            self._sagemaker_client = session.client("sagemaker")
        return self._sagemaker_client

    @property
    def sagemaker_runtime(self) -> Any:
        """Get SageMaker Runtime client for inference."""
        if self._sagemaker_runtime_client is None:
            session = self._get_boto3_session()
            self._sagemaker_runtime_client = session.client("sagemaker-runtime")
        return self._sagemaker_runtime_client

    @property
    def logs(self) -> Any:
        """Get CloudWatch Logs client."""
        if self._logs_client is None:
            session = self._get_boto3_session()
            self._logs_client = session.client("logs")
        return self._logs_client

    @property
    def iam(self) -> Any:
        """Get IAM client."""
        if self._iam_client is None:
            session = self._get_boto3_session()
            self._iam_client = session.client("iam")
        return self._iam_client

    def get_sagemaker_session(self) -> Any:
        """
        Get SageMaker Session for high-level operations.

        Returns:
            SageMaker Session object

        Raises:
            ProviderError: If sagemaker SDK is not installed
        """
        try:
            from sagemaker.core.helper.session_helper import Session

            boto_session = self._get_boto3_session()
            return Session(boto_session=boto_session)
        except ImportError:
            raise ProviderError(
                "sagemaker",
                "sagemaker SDK is not installed. Install with: pip install --upgrade hf-cloud",
            )

    def verify_credentials(self) -> bool:
        """Verify AWS credentials are configured.

        Returns:
            True if credentials are valid

        Raises:
            AuthenticationError: If credentials are invalid
        """
        try:
            self.sagemaker.list_endpoints(MaxResults=1)
            return True
        except Exception as e:
            error_msg = str(e)
            if "credentials" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise AuthenticationError(
                    "sagemaker",
                    "AWS credentials not configured or invalid",
                    details="Run 'aws configure' or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
                )
            raise ProviderError("sagemaker", f"Failed to verify credentials: {e}")

    def get_execution_role(self, role_name: Optional[str] = None) -> str:
        """Resolve and inspect an explicit, configured, or SDK-discovered execution role."""
        role_name = role_name or os.environ.get("SAGEMAKER_ROLE_ARN")
        if role_name is None:
            try:
                from sagemaker.core.helper.session_helper import get_execution_role

                try:
                    role_name = get_execution_role(sagemaker_session=self.get_sagemaker_session())
                except ValueError:
                    role_name = "sagemaker_execution_role"
            except Exception as error:
                raise ProviderError(
                    "sagemaker",
                    "Could not determine SageMaker execution role",
                    details="Provide --role-name NAME or --role ARN, or configure a default execution role",
                ) from error
        return self.validate_execution_role(role_name)

    def validate_execution_role(self, identifier: str) -> str:
        """Fetch IAM role metadata and reject invalid service trust before resource creation."""
        if not isinstance(identifier, str) or not identifier:
            raise ProviderError("sagemaker", "Execution role is empty; provide --role-name NAME or --role ARN")
        expected_arn = None
        if identifier.startswith("arn:"):
            if not re.fullmatch(r"arn:[^:]+:iam::\d{12}:role/.+", identifier):
                raise ProviderError("sagemaker", "Invalid execution role ARN", details=identifier)
            expected_arn = identifier
            name = identifier.rsplit("/", 1)[-1]
        else:
            name = identifier
        if not re.fullmatch(r"[\w+=,.@-]{1,64}", name, flags=re.ASCII):
            raise ProviderError(
                "sagemaker",
                "Invalid execution role name",
                details="Use a role name without its path, or pass the full ARN with --role",
            )
        try:
            role = self.iam.get_role(RoleName=name)["Role"]
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
            if code == "NoSuchEntity":
                message = f"Execution role '{name}' does not exist in the current AWS account"
            elif code in {"AccessDenied", "AccessDeniedException", "UnauthorizedOperation"}:
                message = f"Cannot validate execution role '{name}': iam:GetRole permission is required"
            else:
                message = f"Cannot load execution role '{name}' from IAM"
            raise ProviderError("sagemaker", message, details=str(error)) from error
        if expected_arn and role["Arn"] != expected_arn:
            raise ProviderError(
                "sagemaker",
                "Execution role ARN does not match the role in the current AWS account",
                details=f"Requested {expected_arn}; IAM returned {role['Arn']}",
            )
        confirmed = validate_role_trust(role, self.region)
        if not confirmed:
            console.print(
                Text(
                    "Execution role trust has conditions that require deployment context; AWS will evaluate them during deployment.",
                    style="yellow",
                )
            )
        return role["Arn"]
