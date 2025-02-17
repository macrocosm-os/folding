import os
import boto3
import asyncio
import datetime
import mimetypes

from typing import Optional, Dict
from abc import ABC, abstractmethod
from folding.utils.logging import logger

from dotenv import load_dotenv

load_dotenv()

S3_CONFIG = {
    "region_name": os.getenv("S3_REGION"),
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "access_key_id": os.getenv("S3_KEY"),
    "secret_access_key": os.getenv("S3_SECRET"),
}


class BaseHandler(ABC):
    """Abstract base class for handlers that manage content storage operations."""

    @abstractmethod
    def put(self):
        """Stores content in a designated storage system. This method must be implemented by subclasses."""
        pass


def create_s3_client(
    region_name: str = S3_CONFIG["region_name"],
    endpoint_url: str = S3_CONFIG["endpoint_url"],
    access_key_id: str = S3_CONFIG["access_key_id"],
    secret_access_key: str = S3_CONFIG["secret_access_key"],
) -> boto3.client:
    """Creates a configured S3 client using the environment variables defined in S3_CONFIG.

    Raises:
        ValueError: If any required S3 configuration parameters are missing.

    Returns:
        boto3.S3.Client: A boto3 S3 client configured for use.
    """

    if not all([region_name, endpoint_url, access_key_id, secret_access_key]):
        raise ValueError("Missing required S3 configuration parameters.")
    logger.info(
        f"Creating S3 client with region: {region_name}, endpoint: {endpoint_url}"
    )

    return boto3.session.Session().client(
        "s3",
        region_name=region_name,
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )


async def upload_to_s3(
    handler: "DigitalOceanS3Handler",
    pdb_location: str,
    simulation_cpt: str,
    validator_directory: str,
    pdb_id: str,
    VALIDATOR_ID: str,
) -> Dict[str, str]:
    """Asynchronously uploads PDB and CPT files to S3 using the specified handler.

    Args:
        handler (BaseHandler): The content handler that will execute the upload.
        pdb_location (str): Path to the PDB file.
        simulation_cpt (str): Path to the CPT file.
        validator_directory (str): Directory where validator-specific files are stored.
        pdb_id (str): Identifier for the PDB entry.
        VALIDATOR_ID (str): Identifier for the validator.

    Returns:
        Dict[str, str]: A dictionary of file types and their corresponding S3 URLs.

    Raises:
        Exception: If any error occurs during file upload.
    """
    try:
        s3_links = {}
        input_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        s3_endpoint = os.getenv("S3_ENDPOINT")
        s3_bucket = os.getenv("S3_BUCKET")
        for file_type in ["pdb", "cpt"]:
            if file_type == "cpt":
                file_path = os.path.join(validator_directory, simulation_cpt)
            else:
                file_path = pdb_location

            location = f"inputs/{pdb_id}/{VALIDATOR_ID}/{input_time}"
            logger.debug(
                f"putting file: {file_path} at {location} with type {file_type}"
            )

            key = await asyncio.to_thread(
                handler.put,
                file_path=file_path,
                location=location,
                public=True,
            )
            s3_links[file_type] = os.path.join(f"{s3_endpoint}/{s3_bucket}/", key)
            await asyncio.sleep(0.10)

        return s3_links

    except Exception as e:
        logger.error(f"Exception during file upload:  {str(e)}")
        raise


async def upload_output_to_s3(
    handler: "DigitalOceanS3Handler",
    output_file: str,
    pdb_id: str,
    miner_hotkey: str,
    VALIDATOR_ID: str,
):
    """Asynchronously uploads output files to S3 using the specified handler.

    Args:
        handler (BaseHandler): The content handler that will execute the upload.
        output_file (str): Path to the output file.
        pdb_id (str): Identifier for the PDB entry.
        miner_hotkey (str): Identifier for the miner.
        VALIDATOR_ID (str): Identifier for the validator.

    Returns:
        str: The S3 URL of the uploaded file.

    Raises:
        Exception: If any error occurs during file upload.
    """
    s3_endpoint = os.getenv("S3_ENDPOINT")
    s3_bucket = os.getenv("S3_BUCKET")

    try:
        output_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        location = os.path.join(
            "outputs", pdb_id, VALIDATOR_ID, miner_hotkey[:8], output_time
        )
        key = await asyncio.to_thread(
            handler.put,
            file_path=output_file,
            location=location,
            public=True,
        )
        return os.path.join(f"{s3_endpoint}/{s3_bucket}/", key)
    except Exception as e:
        logger.error(f"Exception during output file upload: {str(e)}")
        raise


class DigitalOceanS3Handler(BaseHandler):
    """Manages DigitalOcean Spaces S3 operations for file storage."""

    def __init__(self, bucket_name: str):
        """Initializes a handler for S3 operations with DigitalOcean Spaces.

        Args:
            bucket_name (str): The name of the S3 bucket.
        """

        self.bucket_name = bucket_name
        self.s3_client = create_s3_client()
        self.custom_mime_types = {
            ".cpt": "application/octet-stream",
            ".pdb": "chemical/x-pdb",
            ".trr": "application/octet-stream",
            ".log": "text/plain",
        }

    def put(
        self,
        file_path: str,
        location: str,
        content_type: Optional[str] = None,
        public: bool = False,
    ):
        """Uploads a file to a specified location in the S3 bucket, optionally setting its access permissions and MIME type.

        Args:
            file_path (str): Local path to the file to upload.
            location (str): Destination path within the bucket.
            content_type (str, optional): MIME type of the file. If None, it's inferred.
            public (bool): Whether to make the file publicly accessible.
            file_type (str, optional): Type of the file, used to determine custom MIME types.

        Returns:
            str: The S3 key of the uploaded file.

        Raises:
            Exception: If the upload fails.
        """

        try:
            file_name = file_path.split("/")[-1]
            key = os.path.join(location, file_name)

            with open(file_path, "rb") as file:
                data = file.read()

            # Infer MIME type
            if not content_type:
                content_type = (
                    self.custom_mime_types.get(
                        file_name[file_name.rfind(".") :]
                    )  # Check custom MIME types first
                    or mimetypes.guess_type(file_path)[
                        0
                    ]  # Fallback to mimetypes library
                    or "application/octet-stream"  # Default to generic binary if no MIME type is found
                )

            # upload file
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                ContentType=content_type,
                ACL="public-read" if public else "private",
            )
            return key
        except Exception as e:
            logger.error(f"handler.put() error: {e}")
            raise
