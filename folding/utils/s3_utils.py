import os
from abc import ABC, abstractmethod
from typing import Optional
import mimetypes
from folding.utils.logging import logger
import boto3
import os 
import asyncio
import datetime

S3_CONFIG = {
    "region_name": os.getenv("S3_REGION"),
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "access_key_id": os.getenv("S3_KEY"),
    "secret_access_key": os.getenv("S3_SECRET"),
    # "bucket_name": os.getenv("S3_BUCKET"),
}

class BaseHandler(ABC):
    """Abstract base class for content handlers.
    
    Defines the interface for content handling operations with get/put operations.
    """

    @abstractmethod
    def put(self):
        """Abstract method to store content."""
        pass

def create_s3_client(
    region_name: str = S3_CONFIG["region_name"],
    endpoint_url: str = S3_CONFIG["endpoint_url"],
    access_key_id: str = S3_CONFIG["access_key_id"],
    secret_access_key: str = S3_CONFIG["secret_access_key"],
) -> boto3.client:  

    """Creates an S3 client"""

    if not all([region_name, endpoint_url, access_key_id, secret_access_key]):
        raise ValueError("Missing required S3 configuration parameters.")
    logger.info(f"Creating S3 client with region: {region_name}, endpoint: {endpoint_url}")

    return boto3.session.Session().client(
        "s3",
        region_name=region_name,
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

async def upload_to_s3(
    handler,
    pdb_location,
    simulation_cpt,
    validator_directory,
    pdb_id,
    VALIDATOR_ID,
):
        try:
            s3_links = {}
            input_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            for file_type in ["pdb", "cpt"]:
                
                if file_type == "cpt":
                    file_path = os.path.join(validator_directory, simulation_cpt)
                else: 
                    file_path = pdb_location 

                location = f"inputs/{pdb_id}/{VALIDATOR_ID}/{input_time}"
                logger.debug(f"putting file: {file_path} at {location} with type {file_type}")
                
                s3_links[file_type] = await asyncio.to_thread(handler.put,
                    file_path=file_path,
                    location=location,
                    public=True, 
                    file_type=file_type
                )
                await asyncio.sleep(0.10)

            return s3_links
        
        except Exception as e:
            logger.error(f"Exception during file upload:  {str(e)}")
            raise

class DigitalOceanS3Handler(BaseHandler):
    """Handles DigitalOcean Spaces S3 operations for content management.

    Manages file content storage operations using DigitalOcean Spaces S3.
    """

    def __init__(self, bucket_name: str):
        """
        Initializes the handler with a bucket name. 
        Args:
            bucket_name (str): The name of the s3 bucket to interact with. 
            custom_mime_types (dict[str, str], optional): A dictionary of custom mime types for specific file extensions. Defaults to None.
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
        file_type:str = None,
    ):
        """
        Upload a file to a specific location in the S3 bucket.
        Args:
            file_path (str): The local path to the file to upload.
            location (str): The destination path within the bucket (e.g., 'inputs/protein/validator').
            content_type (str, optional): The MIME type of the file. If not provided, inferred from file extension.
            public (bool): Whether to make the uploaded file publicly accessible. Defaults to False.
        """

        try:
            file_name = file_path.split("/")[-1]
            key = f"{location}/{file_name}"


            with open(file_path, "rb") as file:
                data = file.read()

            # Infer MIME type 
            if not content_type:
                content_type = (
                    self.custom_mime_types.get(file_name[file_name.rfind(".") :])  # Check custom MIME types first
                    or mimetypes.guess_type(file_path)[0]  # Fallback to mimetypes library
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