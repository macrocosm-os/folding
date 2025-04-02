from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends, Request
from http import HTTPStatus
import pickle
import os
import requests
from loguru import logger
from folding_api.schemas import PDBSearchResponse, PDB, PDBInfoResponse
from folding_api.auth import APIKey, get_api_key
from bs4 import BeautifulSoup

router = APIRouter()

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
