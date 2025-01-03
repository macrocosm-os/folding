import os
from folding.utils.logger import logger
from atom.handlers.handler import create_s3_client, S3Handler
from typing import Optional, Union 
import mimetypes


S3_CONFIG = {
    "region_name": os.getenv("S3_REGION"),
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "access_key_id": os.getenv("S3_KEY"),
    "secret_access_key": os.getenv("S3_SECRET"),
    # "bucket_name": os.getenv("S3_BUCKET"),
}

class DigitalOceanS3Handler(S3Handler):
    """
    Handles DigitalOcean Spaces S3 operations for content management.
    Manages file content retrieval and storage operations using DigitalOcean Spaces S3.
    """

    def __init__(self, bucket_name: str, s3_client=None):
        """
        Initializes the handler with a bucket name and an S3 client.

        Args:
            bucket_name (str): The name of the S3 bucket to interact with.
            s3_client (boto3.client, optional): The S3 client to interact with the bucket. Defaults to None.
        """

        s3_client = s3_client or create_s3_client(**S3_CONFIG)

        super().__init__(bucket_name=bucket_name, s3_client=s3_client)

        # Set MIME types specific to DigitalOceanS3Handler
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
    ) -> Union[str, bool]:
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

            # Infer MIME type from file extension if not provided
            if not content_type:
                content_type = (
                    self.custom_mime_types.get(file_name[file_name.rfind(".") :])  # Check custom MIME types first
                    or mimetypes.guess_type(file_path)[0]  # Fallback to mimetypes library
                    or "application/octet-stream"  # Default to generic binary if no MIME type is found
                )

            
            # upload the file to the bucket. 
            self.s3_client.put_object(
                Bucket=self.bucket_name, 
                Key=key, 
                Body=data,
                ContentType=content_type,
                ACL="public-read" if public else "private",
            )

            if public:
                return key
            else:
                return True

        except FileNotFoundError as e:
            raise e
            # return False
        except Exception as e:
            raise e
            # return False 
        
def upload_to_s3(
    handler,
    pdb_file_location,
    cpt_file_location,
    pdb_id,
    input_time,
    VALIDATOR_ID,
):
    try:
        # upload pdb file to s3
        s3_pdb_link = handler.put(
            file_path=pdb_file_location,
            location=f"inputs/{pdb_id}/{VALIDATOR_ID}/{input_time}",
            content_type=None,
            public=True,
        )
        logger.info(f"Uploaded {pdb_id} pdb file to s3")
        # upload checkpoint file to s3
        s3_cpt_link = handler.put(
            file_path=cpt_file_location,
            location=f"inputs/{pdb_id}/{VALIDATOR_ID}/{input_time}",
            content_type=None,
            public=True,
        )
        logger.info(f"Uploaded {pdb_id} cpt to s3")
    except Exception as e:
        logger.error(f"Failed to upload files to S3: {e}")
        raise e

    return s3_pdb_link, s3_cpt_link
