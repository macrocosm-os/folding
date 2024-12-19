from abc import ABC, abstractmethod
from typing import Any 
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
        Initializes the handler with a bucet name and an s3 client. 

        Args:
        bucket_name (str): The name of the s3 bucket to interact with. 
        s3_client (S3Client): The s3 client to interact with the bucket.
        """

        self.bucket_name = bucket_name
        self.s3_client = s3_client
    
    def put(
        self, 
        file_path: str, 
        pdb_id: str,
        validator_id: str,
        folder: str = "inputs",
        public: bool = False,
    ):
        """Put a file into the s3 bucket under the correct directory structure (assuming you have access to the bucket).
        
        Args:
            file_path (str): The local path to the file to upload.
            pdb_id (str): The PDB identifier for the protein.
            validator_id (str): The identifier for the validator uploading the file.
            folder (str): The top-level folder (e.g., "inputs" or "outputs"). Defaults to "inputs".
            public (bool): Whether to make the uploaded file publicly accessible. Defaults to False.
        """

        try:
            file_name = file_path.split("/")[-1]
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
            key = f"{folder}/{pdb_id}/{validator_id}/{timestamp}/{file_name}"

            print(f"Uploading file to bucket at '{key}'.")

            with open(file_path, "rb") as file:
                data = file.read()

            custom_mime_types = {
                ".cpt": "application/octet-stream",
                ".pdb": "chemical/x-pdb",
                ".trr": "application/octet-stream",
                ".log": "text/plain",
            }

            file_extension = file_name[file_name.rfind(".") :]
            content_type = custom_mime_types.get(
                file_extension, 
                mimetypes.guess_type(file_path)[0] or "application/octet-stream"
                )
            
            # upload the file to the bucket. 
            self.s3_client.s3_client.put_object(
                Bucket=self.bucket_name, 
                Key=key, 
                Body=data,
                ContentType=content_type,
                ACL="public-read" if public and folder == "inputs" else "private",
            )

            print(
                f"File '{file_name}' successfully uploaded to bucket '{self.bucket_name}' at '{key}' with content type '{content_type}'."
            )

        except FileNotFoundError:
            print(f"File '{file_path}' not found. Ensure the path is correct.")
            raise
        except Exception as e:
            print(f"An error occurred while uploading the file:{e}")
            raise 

        
