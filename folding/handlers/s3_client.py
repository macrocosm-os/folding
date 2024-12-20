import boto3


class S3Client:
    """Creates an S3 client"""

    def __init__(self, region_name: str, endpoint_url: str, access_key_id: str, secret_access_key: str):
        """
        Initializes the S3 client
        Args:
        region_name (str): The region name
        endpoint_url (str): The endpoint URL
        access_key_id (str): The access key ID
        secret_access_key (str): The secret access key
        """

        self.s3_client = boto3.session.Session().client(
            "s3",
            region_name=region_name,
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
