import pytest
from unittest.mock import patch, MagicMock
from folding.utils.s3_utils import (
    create_s3_client,
    DigitalOceanS3Handler,
    upload_to_s3,
    upload_output_to_s3,
)


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


def test_handler_initialization():
    with patch("folding.utils.s3_utils.create_s3_client") as mocked_client:
        handler = DigitalOceanS3Handler("test_bucket")
        assert handler.bucket_name == "test_bucket"
        mocked_client.assert_called_once()


@pytest.mark.asyncio
async def test_upload_to_s3_success(mocker):
    handler = mocker.MagicMock(spec=DigitalOceanS3Handler)
    mocker.patch("os.path.join", return_value="file_path")
    mocker.patch("asyncio.to_thread", return_value="key")
    result = await upload_to_s3(
        handler=handler,
        pdb_location="pdb_location",
        simulation_cpt="simulation_ct",
        validator_directory="validator_directory",
        pdb_id="pdb_id",
        VALIDATOR_ID="VALIDATOR_ID",
    )
    assert result == {"pdb": "file_path", "cpt": "file_path"}


@pytest.mark.asyncio
async def test_upload_output_to_s3(mocker):
    handler = mocker.MagicMock(spec=DigitalOceanS3Handler)
    mocker.patch("os.path.join", return_value="path")
    mocker.patch("asyncio.to_thread", return_value="key")
    result = await upload_output_to_s3(
        handler=handler,
        output_file="output_file",
        pdb_id="pdb_id",
        miner_hotkey="miner_hotkey",
        VALIDATOR_ID="VALIDATOR_ID",
    )
    assert result == "path"
