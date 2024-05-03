import time
import pandas as pd
import plotly.express as px
import bittensor as bt
import os


def update_control_file(stage):
    control_file_path = (
        "/path/to/your/directory/current_stage.txt"  # Update this path as needed
    )
    with open(control_file_path, "w") as file:
        file.write(stage)


def auto_plot(
    data_directory,
    stop_event,
    file_name="potential.xvg",
    plot_file="potential_energy_plot.png",
):
    while not stop_event.is_set():
        try:
            file_path = os.path.join(data_directory, file_name)
            df = pd.read_csv(
                file_path, skiprows=24, sep="\s+", names=["time", "potential_energy"]
            )
            fig = px.line(
                df, x="time", y="potential_energy", title="Total Energy vs Time (Step)"
            )
            bt.logging.info("Energy file found, generating plot")
            fig.write_image(os.path.join(data_directory, plot_file))

            # Pause for 10 seconds
            time.sleep(10)
        except FileNotFoundError:
            bt.logging.error(f"File not found: {file_path}")
            time.sleep(10)  # Retry after 10 seconds
        except Exception as e:
            bt.logging.error(f"An error occurred: {e}")
            continue
    bt.logging.info("Plotting thread is stopping as requested.")


if __name__ == "__main__":
    auto_plot("/path/to/your/data")
