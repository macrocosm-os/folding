import os 
from typing import List, Dict
from collections import defaultdict

from folding.utils.logger import logger
from folding.utils.ops import save_pdb 
from folding.base.simulation import OpenMMSimulation

import umap
import numpy as np
from sklearn.preprocessing import StandardScaler

def checkpoint_to_pdb(simulation: OpenMMSimulation, checkpoint_path: str, output_filename: str):
    """Convert a checkpoint file to a PDB file.
    
    Args:
        simulation: The simulation object that has already been loaded with the system config.
        checkpoint_path: The path to the checkpoint file.
        output_filename: The filename of the output PDB file. (needs to include the .pdb extension)
    """
    output_path = os.path.join(os.path.dirname(checkpoint_path), output_filename)

    # load the checkpoint file
    simulation.loadCheckpoint(checkpoint_path)
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()

    save_pdb(positions, simulation.topology, output_path)

def extract_coordinates(pdb_paths: Dict[str, str]) -> Dict[str, List[float]]:
    """Extract the coordinates from a PDB file.
    
    Args:
        pdb_paths: A dictionary of PDB file paths, with the key being the name of the PDB file.

    Returns:
        coordinates: A dictionary of coordinates for each PDB file.
    """
    coordinates = defaultdict(list)

    for name, pdb_path in pdb_paths.items():
        try:
            with open(pdb_path, "r") as f:  
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coordinates[name].extend([x, y, z]) #sinlge long list. 
        except Exception as e:
            logger.error(f"Error extracting coordinates from {pdb_path}: {e}")
            continue

    return coordinates

def embed_pdbs(coordinates: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
    """Embed a set of PDB files using UMAP.
    
    Args:
        coordinates: A dictionary of coordinates for each PDB file.

    Returns:
        embeddings: A numpy array of embeddings for each PDB file.
    """
    coordinate_data = []
    for key in coordinates.keys():
        coordinate_data.append(coordinates[key])
    coordinate_data = np.array(coordinate_data)

    scaler = StandardScaler()
    coordinate_data = scaler.fit_transform(coordinate_data)

    logger.info(f"Fitting UMAP with {coordinate_data.shape[1]} dimensions")
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings = reducer.fit_transform(coordinate_data)

    embeddings_dict = {
        key: embedding for key, embedding in zip(coordinates.keys(), embeddings)
    }

    return embeddings_dict

def visualize_embeddings(embeddings: Dict[str, np.ndarray], image_output_path: str):
    """Visualize the embeddings using UMAP.
    
    Args:
        embeddings: A numpy array of embeddings for each PDB file.
        output_filename: The filename of the output PDB file.
    """

    import pandas as pd 
    import plotly.express as px

    df = pd.DataFrame(embeddings).T
    df.columns = ["umap_1", "umap_2"]

    fig = px.scatter(
        df, 
        x="umap_1", y="umap_2", 
        color=df.index, 
        title = "UMAP of Miner Conformational Manifold", 
        height = 800, 
        width = 1200,
    )
    fig.write_image(image_output_path)
