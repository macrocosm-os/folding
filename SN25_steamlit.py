import bittensor as bt
import streamlit as st
import datetime

import pickle as pkl

# load data from a pkl file
DATA_PATH = "/Users/mccrinbc/Macrocosmos/folding/pdb_ids.pkl"
PDB_IDS = pkl.load(open(DATA_PATH, "rb"))

# Set page configuration for wider layout
st.set_page_config(
    page_title="Molecular Simulation Dashboard",
    layout="wide",  # Use wide layout for more horizontal space
    initial_sidebar_state="collapsed",  # Start with sidebar collapsed
)

# Initialize session state for storing simulation history
if "simulation_history" not in st.session_state:
    st.session_state.simulation_history = []

# Initialize session state for selected option if not exists
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None

# Set page title
st.title("Molecular Simulation Dashboard")

# Create main sections side by side (parameter selection and parameter summary)
main_cols = st.columns([0.65, 0.02, 0.33])  # Left column, divider, right column

# Parameter Selection Section (Left Column)
with main_cols[0]:
    # 1. Search bar with more options for demonstration
    search_options = PDB_IDS["rcsb"]["pdbs"]

    # Create a Google-like search box with proper styling
    st.markdown(
        """
    <style>
    /* Search container styling */
    .search-container {
        margin-bottom: 20px;
    }
    
    /* Google-like search results */
    .search-result {
        padding: 12px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .search-result-title {
        color: #FFFFFF;
        font-size: 18px;
        font-weight: 400;
        margin-bottom: 5px;
        cursor: pointer;
    }
    
    .search-result-title:hover {
        text-decoration: underline;
    }
    
    .search-result-description {
        color: #545454;
        font-size: 14px;
    }
    
    /* Button styling */
    .selected-badge {
        background-color: #e7f5ff;
        color: #0366d6;
        padding: 3px 8px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
        margin-left: 10px;
    }
    
    /* Custom search input */
    .stTextInput > div > div > input {
        font-size: 16px;
        padding: 12px 15px;
        border-radius: 24px;
        border: 1px solid #dfe1e5;
        box-shadow: none;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4285f4;
        box-shadow: 0 1px 6px rgba(32, 33, 36, 0.28);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create a search bar that looks more like Google
    st.write("### Search")
    search_query = st.text_input(
        "",
        key="search_query",
        placeholder="Search for a simulation configuration...",
        label_visibility="collapsed",
    )

    # Filter options based on search query (case-insensitive)
    if search_query:
        filtered_options = [
            opt for opt in search_options if search_query.lower() in opt.lower()
        ]
    else:
        filtered_options = []  # Don't show options until user starts typing

    # Function to handle option selection
    def select_option(option):
        st.session_state.selected_option = option

    # Show filtered results below search bar in Google-like format
    if search_query and not filtered_options:
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        st.warning(
            "Your search did not match any options. Try with different keywords."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    elif filtered_options:
        # Show number of results
        if search_query:
            st.markdown(
                f"<p style='color: #70757a; font-size: 14px; margin-bottom: 10px;'>About {len(filtered_options)} results</p>",
                unsafe_allow_html=True,
            )

        # Display each result in a Google-like format
        for option in filtered_options:
            # Create a unique key for this option
            option_key = f"opt_{option.replace(' ', '_')}"

            # Create the HTML for this search result
            is_selected = st.session_state.selected_option == option
            selected_badge = (
                "<span class='selected-badge'>Selected</span>" if is_selected else ""
            )

            # Highlight matching part of the text
            if search_query and search_query.lower() in option.lower():
                start_idx = option.lower().find(search_query.lower())
                end_idx = start_idx + len(search_query)
                highlighted_option = f"{option[:start_idx]}<strong>{option[start_idx:end_idx]}</strong>{option[end_idx:]}"
            else:
                highlighted_option = option

            # Render the search result
            st.markdown(
                f"""
            <div class="search-result" id="{option_key}">
                <div class="search-result-title">{highlighted_option} {selected_badge}</div>
                <div class="search-result-description">
                    Simulation configuration with predefined parameters.
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # We need a way to select options since HTML clicks don't work directly with Streamlit
            cols = st.columns([0.85, 0.15])
            with cols[1]:
                if not is_selected:
                    if st.button(
                        "Select", key=f"btn_{option}", use_container_width=True
                    ):
                        select_option(option)
                        st.session_state.update_page = True
                else:
                    st.button(
                        "Selected ✓",
                        key=f"btn_{option}",
                        disabled=True,
                        use_container_width=True,
                    )

    # Display currently selected option
    selected_option = st.session_state.selected_option
    if selected_option:
        # Add clear selection button
        if st.button("Clear Selection", key="clear_btn", type="secondary"):
            st.session_state.selected_option = None
            selected_option = None
            st.session_state.update_page = True

    # Check if update flag is set and clear it
    if st.session_state.get("update_page", False):
        st.session_state.update_page = False
        st.rerun()  # Only call rerun once at the end

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
