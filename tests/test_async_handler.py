import pytest
from folding.handlers.aio_s3_client import create_s3_client 
from folding.handlers.aiohandlers import DigitalOceanS3Handler 
from aiobotocore.client import AioBaseClient
from unittest.mock import AsyncMock, patch, MagicMock
import aioboto3

@pytest.mark.asyncio
async def test_create_s3_client_success(mocker):
    mock_session = mocker.patch("folding.handlers.aio_s3_client.aioboto3.Session")
    mock_client = mocker.Mock(spec=AioBaseClient)
    mock_session.return_value.client.return_value = mock_client 

    client = create_s3_client(
        region_name = "mock-region",
        endpoint_url = "https://mock-endpoint.com",
        access_key_id = "mock-access-key",
        secret_access_key="mock-secret-key"
    )

    assert client == mock_client 
    mock_session.return_value.client.assert_called_once_with(
        "s3",
        region_name="mock-region",
        endpoint_url="https://mock-endpoint.com",
        aws_access_key_id="mock-access-key",
        aws_secret_access_key="mock-secret-key",
    )

@pytest.mark.asyncio
async def test_dos3handler_put_success_with_tmp_file(tmp_path, mocker):
    # Create a temporary file
    temp_file = tmp_path / "file.txt"
    temp_file.write_bytes(b"mock-data")

    # Mock aioboto3.Session and its client method to return a mock s3 client
    mock_s3_client = AsyncMock()
    mock_s3_client.put_object = AsyncMock()

    # Create a mock session that returns a mock client from __aenter__
    mock_session = MagicMock()
    mock_session.client.return_value.__aenter__.return_value = mock_s3_client

    # Patch 'aioboto3.Session' to use our mock session
    with patch('aioboto3.Session', return_value=mock_session):
        # Create handler instance
        handler = DigitalOceanS3Handler(bucket_name="mock-bucket", s3_session=aioboto3.Session())

        # Call the put method
        await handler.put(
            file_path=str(temp_file),
            location="mock-location",
            content_type="text/plain",
            public=True,
        )

    # Assertions
    mock_s3_client.put_object.assert_called_once_with(
        Bucket="mock-bucket",
        Key="mock-location/file.txt",
        Body=b"mock-data",
        ContentType="text/plain",
        ACL="public-read",
    )
