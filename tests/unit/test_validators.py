"""Tests for validators."""

import pytest

from hf_cloud.utils.validators import (
    validate_deployment_name,
    validate_instance_type,
    validate_model_id,
    validate_region,
)


class TestValidateModelId:
    """Tests for model ID validation."""

    def test_valid_simple_model_id(self):
        """Test valid simple model IDs."""
        is_valid, error = validate_model_id("gpt2")
        assert is_valid
        assert error == ""

    def test_valid_namespaced_model_id(self):
        """Test valid namespaced model IDs."""
        is_valid, error = validate_model_id("meta-llama/Llama-2-7b-hf")
        assert is_valid
        assert error == ""

    def test_valid_model_with_dots(self):
        """Test model ID with dots."""
        is_valid, error = validate_model_id("bert-base-uncased")
        assert is_valid

    def test_empty_model_id(self):
        """Test empty model ID."""
        is_valid, error = validate_model_id("")
        assert not is_valid
        assert "empty" in error.lower()

    def test_model_id_too_long(self):
        """Test model ID that's too long."""
        is_valid, error = validate_model_id("a" * 201)
        assert not is_valid
        assert "too long" in error.lower()

    def test_model_id_double_slash(self):
        """Test model ID with double slash."""
        is_valid, error = validate_model_id("user//model")
        assert not is_valid

    def test_model_id_leading_slash(self):
        """Test model ID with leading slash."""
        is_valid, error = validate_model_id("/model")
        assert not is_valid


class TestValidateDeploymentName:
    """Tests for deployment name validation."""

    def test_valid_sagemaker_name(self):
        """Test valid SageMaker deployment name."""
        is_valid, error = validate_deployment_name("my-endpoint-123", "sagemaker")
        assert is_valid

    def test_sagemaker_name_too_long(self):
        """Test SageMaker name that's too long."""
        is_valid, error = validate_deployment_name("a" * 64, "sagemaker")
        assert not is_valid
        assert "63" in error

    def test_sagemaker_name_invalid_chars(self):
        """Test SageMaker name with invalid characters."""
        is_valid, error = validate_deployment_name("my_endpoint", "sagemaker")
        assert not is_valid

    def test_valid_azure_name(self):
        """Test valid Azure deployment name."""
        is_valid, error = validate_deployment_name("my-endpoint", "azure")
        assert is_valid

    def test_azure_name_too_short(self):
        """Test Azure name that's too short."""
        is_valid, error = validate_deployment_name("ab", "azure")
        assert not is_valid

    def test_empty_name(self):
        """Test empty deployment name."""
        is_valid, error = validate_deployment_name("", "sagemaker")
        assert not is_valid


class TestValidateInstanceType:
    """Tests for instance type validation."""

    def test_valid_sagemaker_instance(self):
        """Test valid SageMaker instance type."""
        is_valid, error = validate_instance_type("ml.g5.xlarge", "sagemaker")
        assert is_valid

    def test_valid_sagemaker_p4d(self):
        """Test valid SageMaker P4D instance."""
        is_valid, error = validate_instance_type("ml.p4d.24xlarge", "sagemaker")
        assert is_valid

    def test_invalid_sagemaker_instance(self):
        """Test invalid SageMaker instance type."""
        is_valid, error = validate_instance_type("m5.xlarge", "sagemaker")
        assert not is_valid

    def test_valid_azure_instance(self):
        """Test valid Azure instance type."""
        is_valid, error = validate_instance_type("Standard_DS3_v2", "azure")
        assert is_valid

    def test_invalid_azure_instance(self):
        """Test invalid Azure instance type."""
        is_valid, error = validate_instance_type("DS3_v2", "azure")
        assert not is_valid

    def test_empty_instance_type(self):
        """Test empty instance type."""
        is_valid, error = validate_instance_type("", "sagemaker")
        assert not is_valid


class TestValidateRegion:
    """Tests for region validation."""

    def test_valid_aws_region(self):
        """Test valid AWS region."""
        is_valid, error = validate_region("us-east-1", "sagemaker")
        assert is_valid

    def test_invalid_aws_region(self):
        """Test invalid AWS region."""
        is_valid, error = validate_region("invalid-region", "sagemaker")
        assert not is_valid

    def test_valid_gcp_region(self):
        """Test valid GCP region."""
        is_valid, error = validate_region("us-central1", "vertex")
        assert is_valid

    def test_empty_region(self):
        """Test empty region."""
        is_valid, error = validate_region("", "sagemaker")
        assert not is_valid
