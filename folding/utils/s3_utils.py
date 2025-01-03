import os
from folding.utils.logger import logger


def upload_to_s3(
    handler,
    pdb_location,
    simulation_cpt,
    validator_directory,
    pdb_id,
    input_time,
    VALIDATOR_ID,
):
    try:
        # upload pdb file to s3
        s3_pdb_link = handler.put(
            file_path=pdb_location,
            location=f"inputs/{pdb_id}/{VALIDATOR_ID}/{input_time}",
            content_type=None,
            public=True,
        )
        logger.info(f"Uploaded pdb file to s3")
        # upload checkpoint file to s3
        s3_cpt_link = handler.put(
            file_path=os.path.join(
                validator_directory.rsplit("/", 1)[0], simulation_cpt
            ),
            location=f"inputs/{pdb_id}/{VALIDATOR_ID}/{input_time}",
            content_type=None,
            public=True,
        )
        logger.info(f"Uploaded checkpoint file to s3")
    except Exception as e:
        logger.error(f"Failed to upload files to S3: {e}")
        raise e

    return s3_pdb_link, s3_cpt_link
