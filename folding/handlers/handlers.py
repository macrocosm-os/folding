from abc import ABC, abstractmethod
from typing import Any, Optional
from folding.handlers.s3_client import S3Client
import mimetypes
import datetime 

class BaseHandler(ABC):
    """Abstract base class for content handlers.
    
    Defines the interface for content handling operations with get/put operations.
    """

    @abstractmethod
    def put(self):
        """Abstract method to store content."""
        pass

class DigitalOceanS3Handler(BaseHandler):
    """Handles DigitalOcean Spaces S3 operations for content management.

    Manages file content retrieval and storage operations using DigitalOcean Spaces S3.
    """

    def __init__(self, bucket_name: str, s3_client: S3Client):
        """
        Initializes the handler with a bucket name and an s3 client. 

        Args:
        bucket_name (str): The name of the s3 bucket to interact with. 
        s3_client (S3Client): The s3 client to interact with the bucket.
        custom_mime_types (dict[str, str], optional): A dictionary of custom mime types for specific file extensions. Defaults to None.
        """

        self.bucket_name = bucket_name
        self.s3_client = s3_client
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

            print(f"Uploading file to S3 with Key: {key}")  # Debugging

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
            self.s3_client.s3_client.put_object(
                Bucket=self.bucket_name, 
                Key=key, 
                Body=data,
                ContentType=content_type,
                ACL="public-read" if public else "private",
            )

            print(
                f"File '{file_name}' successfully uploaded to '{key}' in bucket '{self.bucket_name}' with content type '{content_type}'."
            )

        except FileNotFoundError:
            print(f"File '{file_path}' not found. Ensure the path is correct.")
            raise
        except Exception as e:
            print(f"An error occurred while uploading the file:{e}")
            raise 

        
