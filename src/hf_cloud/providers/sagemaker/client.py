
"""AWS SageMaker client wrapper."""

from typing import Any, Optional

from rich.console import Console

from hf_cloud.core.config import Config
from hf_cloud.core.exceptions import AuthenticationError, ProviderError

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
                "boto3 is not installed. Install with: pip install hf-cloud[sagemaker]",
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
                    "boto3 is not installed. Install with: pip install hf-cloud[sagemaker]",
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
                "sagemaker SDK is not installed. Install with: pip install hf-cloud[sagemaker]",
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
        """Get SageMaker execution role ARN.

        Args:
            role_name: Optional explicit role ARN or name

        Returns:
            Role ARN

        Raises:
            ProviderError: If role cannot be determined
        """
        # If a full ARN is provided, return it
        if role_name and role_name.startswith("arn:aws:iam::"):
            return role_name

        if role_name:
            # Try to find a SageMaker role by name or search for one
            try:
                iam = self.iam

                # If a role name was provided, try to get it
                if role_name:
                    try:
                        role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
                        console.print(f"[green]Using IAM role: {role_arn}[/green]")
                        return role_arn
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            # Try to get role from SageMaker SDK (works in SageMaker notebooks)
            try:
                from sagemaker.core.helper.session_helper import get_execution_role

                role = get_execution_role(sagemaker_session=self.get_sagemaker_session())
                console.print(f"[green]Using SageMaker execution role: {role}[/green]")
                return role
            except Exception:
                pass

            # Try to find a SageMaker role by name or search for one
            try:
                iam = self.iam

                # If a role name was provided, try to get it
                if role_name:
                    try:
                        role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
                        console.print(f"[green]Using IAM role: {role_arn}[/green]")
                        return role_arn
                    except Exception:
                        pass
            except Exception:
                pass

        raise ProviderError(
            "sagemaker",
            "Could not determine SageMaker execution role",
            details="Provide --role with an ARN or role name, or ensure a SageMaker execution role exists",
        )
