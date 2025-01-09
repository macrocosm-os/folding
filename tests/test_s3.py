import pytest
from unittest.mock import patch
from folding.utils.s3_utils import create_s3_client

@patch("boto3.session.Session.client")
def test_create_s3_client(mock_boto_client):
    mock_s3 = mock_boto_client.return_value

    client = create_s3_client(
        region_name="mock-region",
        endpoint_url="http://mock-endpoint",
        access_key_id="mock-access-key",
        secret_access_key="mock-secret-key",
    )

    mock_boto_client.assert_called_once_with(
        "s3",
        region_name="mock-region",
        endpoint_url="http://mock-endpoint",
        aws_access_key_id="mock-access-key",
        aws_secret_access_key="mock-secret-key",
    )

    assert client == mock_s3
