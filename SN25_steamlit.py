import bittensor as bt
import streamlit as st
import datetime

# Set page configuration for wider layout
st.set_page_config(
    page_title="Molecular Simulation Dashboard",
    layout="wide",  # Use wide layout for more horizontal space
    initial_sidebar_state="collapsed",  # Start with sidebar collapsed
)

# Initialize session state for storing simulation history
if "simulation_history" not in st.session_state:
    st.session_state.simulation_history = []

# Set page title
st.title("Molecular Simulation Dashboard")

# Create main sections side by side (parameter selection and parameter summary)
main_cols = st.columns([0.65, 0.02, 0.33])  # Left column, divider, right column

# Parameter Selection Section (Left Column)
with main_cols[0]:
    # 1. Search bar with options
    search_options = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
    selected_option = st.selectbox(
        "Search for an option",
        search_options,
        index=None,
        placeholder="Search...",
    )

    if selected_option:
        st.write(f"You selected: {selected_option}")

    # Create a layout with columns within the left section
    col1, col2 = st.columns(2)

    with col1:
        # 2. Temperature slider
        temperature = st.slider(
            "Temperature (K)", min_value=200, max_value=400, value=300, step=1
        )
        st.write(f"Selected temperature: {temperature} K")

        # 3. Friction slider
        friction = st.slider(
            "Friction", min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )
        st.write(f"Selected friction: {friction}")

    with col2:
        # 4. Pressure slider
        pressure = st.slider(
            "Pressure (atm)", min_value=1.0, max_value=2.0, value=1.5, step=0.1
        )
        st.write(f"Selected pressure: {pressure} atm")

        # 5. Forcefield dropdown
        forcefield_options = ["CHARMM", "AMBER", "OPLS"]
        forcefield = st.selectbox("Select Forcefield", forcefield_options)
        st.write(f"Selected forcefield: {forcefield}")

        # 6. Water dropdown
        water_options = ["TIP3P", "SPC/E", "TIP4P"]
        water_model = st.selectbox("Select Water Model", water_options)
        st.write(f"Selected water model: {water_model}")

    # Simulation name input - with the default set to selected_option if available
    simulation_name = st.text_input(
        "Simulation Name",
        value=selected_option if selected_option else "",
        placeholder="Enter a name for this simulation run...",
    )

    # Add a run button
    run_simulation = st.button("Run Simulation")

# Vertical divider (middle column)
with main_cols[1]:
    # Create a vertical divider using CSS
    st.markdown(
        """
        <div style="background-color: #e0e0e0; width: 2px; height: 100%; margin: 0 auto;"></div>
        """,
        unsafe_allow_html=True,
    )

# Parameter Summary Section (Right Column)
with main_cols[2]:
    st.subheader("Parameter Summary")

    # Create a styled summary container with all parameters
    summary_container = st.container()
    with summary_container:
        # Display parameters in a clean format with background
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)

        st.markdown("**Selected Option:**")
        st.markdown(f"```{selected_option if selected_option else 'None'}```")

        st.markdown("**Temperature:**")
        st.markdown(f"```{temperature} K```")

        st.markdown("**Friction:**")
        st.markdown(f"```{friction}```")

        st.markdown("**Pressure:**")
        st.markdown(f"```{pressure} atm```")

        st.markdown("**Forcefield:**")
        st.markdown(f"```{forcefield}```")

        st.markdown("**Water Model:**")
        st.markdown(f"```{water_model}```")

        st.markdown("</div>", unsafe_allow_html=True)

# Process the simulation run if button was clicked
if run_simulation:
    # Use simulation_name if provided, otherwise use selected_option or "Unnamed Simulation"
    sim_name = (
        simulation_name
        if simulation_name
        else (selected_option if selected_option else "Unnamed Simulation")
    )

    # Create a dictionary of the current parameters
    current_params = {
        "Simulation Name": sim_name,
        "Selected Option": selected_option,
        "Temperature": f"{temperature} K",
        "Friction": friction,
        "Pressure": f"{pressure} atm",
        "Forcefield": forcefield,
        "Water Model": water_model,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Add to history
    st.session_state.simulation_history.append(current_params)

    st.success(f"Simulation '{sim_name}' started with the following parameters:")
    st.json(current_params)

# Horizontal divider before history section
st.markdown("---")

# History Section (Below both parameter selection and summary)
st.subheader("Simulation History")

history_container = st.container()
with history_container:
    if not st.session_state.simulation_history:
        st.info("No simulations run yet. Submit a simulation to see its history.")
    else:
        # Create a grid layout for history entries
        history_cols = st.columns(3)

        # Display history in reverse order (newest first)
        for i, params in enumerate(reversed(st.session_state.simulation_history)):
            # Distribute history entries across columns
            col_index = i % 3

            with history_cols[col_index]:
                sim_title = params.get(
                    "Simulation Name",
                    f"Simulation #{len(st.session_state.simulation_history) - i}",
                )
                with st.expander(sim_title):
                    # Create two columns for parameters within the expander
                    h_col1, h_col2 = st.columns(2)

                    with h_col1:
                        st.markdown("**Selected Option:**")
                        st.markdown(
                            f"```{params['Selected Option'] if params['Selected Option'] else 'None'}```"
                        )

                        st.markdown("**Temperature:**")
                        st.markdown(f"```{params['Temperature']}```")

                        st.markdown("**Friction:**")
                        st.markdown(f"```{params['Friction']}```")

                    with h_col2:
                        st.markdown("**Pressure:**")
                        st.markdown(f"```{params['Pressure']}```")

                        st.markdown("**Forcefield:**")
                        st.markdown(f"```{params['Forcefield']}```")

                        st.markdown("**Water Model:**")
                        st.markdown(f"```{params['Water Model']}```")

                    st.caption(f"Run on: {params.get('Timestamp', 'Unknown time')}")
