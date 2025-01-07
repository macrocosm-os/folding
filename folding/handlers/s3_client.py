import boto3
import os 
from folding.utils.logger import logger

S3_CONFIG = {
    "region_name": os.getenv("S3_REGION"),
    "endpoint_url": os.getenv("S3_ENDPOINT"),
    "access_key_id": os.getenv("S3_KEY"),
    "secret_access_key": os.getenv("S3_SECRET"),
    # "bucket_name": os.getenv("S3_BUCKET"),
}

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
