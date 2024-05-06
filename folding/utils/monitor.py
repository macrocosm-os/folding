import os
import pandas as pd
import plotly.express as px
import logging

def auto_plot(data_location: str, file_name: str, output_file: str):
    try:
        file_path = os.path.join(data_location, file_name)
        df = pd.read_csv(file_path, skiprows=24, sep="\s+", names=["time", "potential_energy"])
        fig = px.line(df, x="time", y="potential_energy", title="Total Energy vs Time (Step)")
        fig.write_image(os.path.join(data_location, output_file))
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")