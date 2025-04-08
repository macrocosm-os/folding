import subprocess
from typing import Optional, Literal
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from http import HTTPStatus
import pickle
import os
import requests
from loguru import logger
from folding.utils.ops import convert_cif_to_pdb
from folding_api.schemas import (
    PDBSearchResponse,
    PDB,
    PDBInfoResponse,
    JobPoolResponse,
    Job,
)
from folding_api.auth import APIKey, get_api_key
from folding_api.utils import query_gjp
import json

router = APIRouter(tags=["Utility Endpoints"])

# Global variables to store PDB data
PDB_DATA = None
PDB_TO_SOURCE = {}
ALL_PDB_IDS = []

# Load PDB data when module is initialized
def load_pdb_data():
    global PDB_DATA, PDB_TO_SOURCE, ALL_PDB_IDS

    try:
        # Load the PDB IDs from the pickle file
        pdb_ids_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pdb_ids.pkl"
        )

        if not os.path.exists(pdb_ids_path):
            logger.error("PDB IDs database not found")
            return False

        with open(pdb_ids_path, "rb") as f:
            PDB_DATA = pickle.load(f)

        # Create a mapping of PDB IDs to their sources
        for src in ["rcsb", "pdbe"]:
            if src in PDB_DATA:
                for pdb_id in PDB_DATA[src]["pdbs"]:
                    PDB_TO_SOURCE[pdb_id] = src
                    ALL_PDB_IDS.append(pdb_id)

        logger.info(f"Loaded {len(ALL_PDB_IDS)} PDB IDs into memory")
        return True
    except Exception as e:
        logger.exception(f"Error loading PDB data: {e}")
        return False


# Initialize the data when the module is loaded
load_pdb_data()


@router.get("/search", response_model=PDBSearchResponse)
async def search_pdb(
    request: Request,
    query: str = Query(..., description="Search query for PDB IDs"),
    limit: int = Query(100, description="Maximum number of results to return"),
    threshold: Optional[int] = Query(
        60, description="Minimum similarity score (0-100) for matching"
    ),
    api_key: APIKey = Depends(get_api_key),
) -> PDBSearchResponse:
    """
    Search for PDB IDs in the database.

    This endpoint searches through the pdb_ids.pkl file and returns a list of matching PDB IDs.
    The search uses simple substring matching to find PDB IDs that contain the query string.
    Results are sorted by position of the match (matches at the beginning rank higher).

    Each PDB ID is returned with its source (rcsb or pdbe).
    """
    try:
        # Check if PDB data is loaded
        if PDB_DATA is None:
            # Try to load the data if it's not already loaded
            if not load_pdb_data():
                raise HTTPException(
                    status_code=HTTPStatus.NOT_FOUND,
                    detail="PDB IDs database not found or could not be loaded",
                )

        # Prepare the search query
        query = query.lower()

        # Find substring matches
        matches = []
        for pdb_id in ALL_PDB_IDS:
            pdb_id_lower = pdb_id.lower()
            if query in pdb_id_lower:
                # Calculate position of match (for sorting)
                position = pdb_id_lower.find(query)
                matches.append((pdb_id, position))

        # Sort by position (matches at the beginning come first)
        matches.sort(key=lambda x: x[1])

        # Apply limit
        limited_matches = matches[:limit]

        # Extract just the PDB IDs
        result_pdb_ids = [pdb_id for pdb_id, _ in limited_matches]

        # Return the results
        return PDBSearchResponse(
            matches=[
                PDB(pdb_id=pdb_id, source=PDB_TO_SOURCE[pdb_id])
                for pdb_id in result_pdb_ids
            ],
            total=len(matches),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during PDB search: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during the search. Please try again later.",
        )


@router.get("/pdb/{pdb_id}", response_model=PDBInfoResponse)
async def get_pdb_info(
    pdb_id: str,
    api_key: APIKey = Depends(get_api_key),
) -> PDBInfoResponse:
    """
    Retrieve detailed information about a PDB structure from RCSB.

    This endpoint queries the RCSB PDB GraphQL API to get information about a specific PDB entry.
    """
    try:
        # Normalize PDB ID
        pdb_id = pdb_id.lower()

        # Check if the PDB ID exists in our database
        if PDB_DATA is None:
            if not load_pdb_data():
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail="PDB database could not be loaded",
                )

        if pdb_id not in PDB_TO_SOURCE:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"PDB ID {pdb_id} not found in database",
            )

        # Fetch details from RCSB GraphQL API
        graphql_url = "https://data.rcsb.org/graphql"

        # Construct GraphQL query
        query = """
        query GetPDBInfo($pdbId: String!) {
          entry(entry_id: $pdbId) {
            struct {
              title
            }
            struct_keywords {
              pdbx_keywords
            }
            polymer_entities {
              rcsb_entity_source_organism {
                scientific_name
              }
              rcsb_entity_host_organism {
                scientific_name
              }
              entity_poly {
                pdbx_seq_one_letter_code_can
                rcsb_sample_sequence_length
              }
            }
            rcsb_entry_info {
              experimental_method
              resolution_combined
            }
          }
        }
        """

        # Make GraphQL request
        response = requests.post(
            graphql_url, json={"query": query, "variables": {"pdbId": pdb_id}}
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"PDB ID {pdb_id} not found in RCSB database",
            )

        result = response.json()

        # Check for GraphQL errors
        if "errors" in result:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"Error retrieving data for PDB ID {pdb_id}: {result['errors'][0]['message']}",
            )

        # Extract data from GraphQL response
        data = result.get("data", {}).get("entry", {})
        if not data:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"No data found for PDB ID {pdb_id}",
            )

        # Extract relevant information
        struct_data = data.get("struct", {})
        molecule_name = struct_data.get("title")
        struct_keywords = data.get("struct_keywords", {})
        classification = struct_keywords.get("pdbx_keywords")
        print(struct_keywords)

        # Extract organism information
        polymer_entities = data.get("polymer_entities", [])
        organism = None
        expression_system = None

        if polymer_entities and len(polymer_entities) > 0:
            # Get organism from first entity
            org_data = polymer_entities[0].get("rcsb_entity_source_organism", [])
            if org_data and len(org_data) > 0:
                organism = org_data[0].get("scientific_name")

            # Get expression system from first entity
            host_data = polymer_entities[0].get("rcsb_entity_host_organism", [])
            if host_data and len(host_data) > 0:
                expression_system = host_data[0].get("scientific_name")

        # Construct and return the PDBInfoResponse
        return PDBInfoResponse(
            pdb_id=pdb_id,
            molecule_name=molecule_name,
            classification=classification,
            organism=organism,
            expression_system=expression_system,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error retrieving PDB info: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving PDB information. Please try again later.",
        )


@router.get("/pdb/{pdb_id}/file")
async def get_pdb_file(
    pdb_id: str,
    input_source=Query(..., description="The source of the PDB file to download"),
):
    """
    Retrieve the PDB file for a given PDB ID.
    First checks the S3 bucket, if not found falls back to RCSB or PDBe sources.
    """
    try:
        # Normalize PDB ID
        pdb_id = pdb_id.lower()

        # First check S3 bucket
        s3_url = f"https://sn25.nyc3.digitaloceanspaces.com/pdb_files/{pdb_id}.pdb"
        s3_response = requests.get(s3_url)
        if s3_response.status_code == 200:
            logger.info(f"Found PDB file {pdb_id} in S3 bucket")
            return s3_response.text

        # If not in S3, proceed with original download methods
        if input_source == "rcsb":
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            r = requests.get(url)
            if r.status_code == 200:
                return r.text
            else:
                raise HTTPException(
                    status_code=HTTPStatus.NOT_FOUND,
                    detail=f"PDB file {pdb_id} not found in database.",
                )

        elif input_source == "pdbe":
            # strip the string of the extension
            substring = pdb_id[1:3]
            temp_dir = os.path.join(os.path.dirname(__file__), "temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            unzip_command = ["gunzip", f"{temp_dir}/{pdb_id}.cif.gz"]

            rsync_command = [
                "rsync",
                "-rlpt",
                "-v",
                "-z",
                f"rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/mmCIF/{substring}/{pdb_id}.cif.gz",
                f"{temp_dir}/",
            ]

            try:
                subprocess.run(rsync_command, check=True)
                subprocess.run(unzip_command, check=True)
                logger.success(f"PDB file {pdb_id} downloaded successfully from PDBe.")

                convert_cif_to_pdb(
                    cif_file=f"{temp_dir}/{pdb_id}.cif",
                    pdb_file=f"{temp_dir}/{pdb_id}.pdb",
                )
                with open(f"{temp_dir}/{pdb_id}.pdb", "r") as file:
                    pdb_text = file.read()
                os.remove(f"{temp_dir}/{pdb_id}.cif")
                os.remove(f"{temp_dir}/{pdb_id}.pdb")
                return pdb_text
            except subprocess.CalledProcessError as e:
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail=f"Failed to download PDB file with ID {pdb_id} from PDBe.",
                )

        else:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Unknown input source: {input_source}",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error retrieving PDB file: {e}")


@router.get("/pdb/{pdb_id}/images")
async def get_pdb_images(
    pdb_id: str,
    api_key: APIKey = Depends(get_api_key),
):
    """
    Retrieve PDB structure image URLs from S3 bucket.
    Returns URLs for both small (200px) and large (800px) versions of the image.
    """
    try:
        # Normalize PDB ID
        pdb_id = pdb_id.lower()

        # Construct image URLs
        base_url = "https://sn25.nyc3.digitaloceanspaces.com/pdb_images"
        small_image_url = f"{base_url}/{pdb_id}_200.png"
        large_image_url = f"{base_url}/{pdb_id}_800.png"

        # Check if images exist
        small_exists = requests.head(small_image_url).status_code == 200
        large_exists = requests.head(large_image_url).status_code == 200

        if not (small_exists or large_exists):
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"No images found for PDB ID {pdb_id}",
            )

        return {
            "small_image": small_image_url if small_exists else None,
            "large_image": large_image_url if large_exists else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error retrieving PDB images: {e}")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving PDB images. Please try again later.",
        )


@router.get("/job_pool", response_model=JobPoolResponse)
async def get_job_pool_status(
    status: Literal["active", "inactive", "failed", "all"],
    api_key: APIKey = Depends(get_api_key),
):
    """
    Retrieve the status of the job pool.
    """
    if status == "active":  # active = 1
        query = "SELECT * FROM jobs WHERE active = 1"
    elif status == "inactive":  # active = 0 and not failed
        query = "SELECT * FROM jobs WHERE active = 0 AND (event NOT LIKE '%\"failed\": true%' OR event IS NULL)"
    elif status == "failed":  # active = 0 and event.failed = true
        query = (
            "SELECT * FROM jobs WHERE active = 0 AND event LIKE '%\"failed\": true%'"
        )
    elif status == "all":
        query = "SELECT * FROM jobs"

    results = query_gjp(query)
    jobs = []
    for result in results:
        if not result:
            continue

        event = json.loads(result.get("event", {}))
        # Determine job status based on active and event data
        if result["active"] == "1":
            job_status = "active"
        elif result["active"] == "0" and event.get("failed", False):
            job_status = "failed"
        elif result["active"] == "0":
            job_status = "inactive"
        else:
            job_status = "inactive"  # Default to inactive for unknown cases

        job = Job(
            id=str(result["id"]),
            type="organic" if result["is_organic"] == 1 else "synthetic",
            job_id=result["job_id"],
            pdb_id=result["pdb_id"],
            created_at=result["created_at"],
            priority=result["priority"],
            validator_hotkey=result["validator_hotkey"],
            best_hotkey=result["best_hotkey"],
            s3_links=json.loads(result["s3_links"]),
            status=job_status,
        )
        jobs.append(job)

    resp = JobPoolResponse(jobs=jobs, total=len(jobs))
    return resp
