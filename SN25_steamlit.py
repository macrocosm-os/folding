import datetime
import pickle as pkl
import streamlit as st
import bittensor as bt
from folding.utils.openmm_forcefields import FORCEFIELD_REGISTRY
import os
import json
import requests
from atom.epistula.epistula import Epistula

# load data from a pkl file
DATA_PATH = "pdb_ids.pkl"
PDB_IDS = pkl.load(open(DATA_PATH, "rb"))
API_ADDRESS = "184.105.5.57:8031"
GJP_ADDRESS = "167.99.209.27:4001"

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

# Initialize session state for pagination
if "page_number" not in st.session_state:
    st.session_state.page_number = 0

# Initialize session state for wallet
if "wallet" not in st.session_state:
    st.session_state.wallet = None

# Initialize session state for wallet configuration
if "wallet_name" not in st.session_state:
    st.session_state.wallet_name = ""
if "wallet_hotkey" not in st.session_state:
    st.session_state.wallet_hotkey = ""

# Initialize session state for forcefield and water model
if "selected_forcefield" not in st.session_state:
    st.session_state.selected_forcefield = None
if "selected_water" not in st.session_state:
    st.session_state.selected_water = None

# Initialize session state for last status update time
if "last_status_update" not in st.session_state:
    st.session_state.last_status_update = {}


def get_wallet_names():
    """Get list of wallet names from ~/.bittensor/wallets/"""
    wallet_dir = os.path.expanduser("~/.bittensor/wallets/")
    if not os.path.exists(wallet_dir):
        return []
    return [
        d for d in os.listdir(wallet_dir) if os.path.isdir(os.path.join(wallet_dir, d))
    ]


def get_hotkeys(wallet_name):
    """Get list of hotkeys for a given wallet name"""
    wallet_dir = os.path.expanduser(f"~/.bittensor/wallets/{wallet_name}/hotkeys")
    if not os.path.exists(wallet_dir):
        return []

    # return all non directories
    return [
        f for f in os.listdir(wallet_dir) if os.path.isfile(os.path.join(wallet_dir, f))
    ]


def create_wallet():
    if st.session_state.wallet_name and st.session_state.wallet_hotkey:
        try:
            st.session_state.wallet = bt.wallet(
                name=st.session_state.wallet_name, hotkey=st.session_state.wallet_hotkey
            )
            st.success(
                f"Wallet configured successfully: {st.session_state.wallet.hotkey.ss58_address}"
            )
        except Exception as e:
            st.error(f"Error configuring wallet: {str(e)}")
            st.session_state.wallet = None


def response_to_dict(response):
    response = response.json()["results"][0]
    if "error" in response.keys():
        raise ValueError(f"Failed to get all PDBs: {response['error']}")
    elif "values" not in response.keys():
        return {}
    columns = response["columns"]
    values = response["values"]
    data = [dict(zip(columns, row)) for row in values]
    return data


# Function to handle option selection
def select_option(option):
    st.session_state.selected_option = option


# Function to display search results with pagination
def display_results(options, search_query, items_per_page=5):
    # Calculate total pages
    total_pages = (len(options) - 1) // items_per_page + 1

    # Ensure page_number is valid
    if st.session_state.page_number >= total_pages:
        st.session_state.page_number = 0

    # Calculate start and end indices for current page
    start_idx = st.session_state.page_number * items_per_page
    end_idx = min(start_idx + items_per_page, len(options))

    # Display only the options for the current page
    current_page_options = options[start_idx:end_idx]

    # Display each option in the current page
    for option in current_page_options:
        # Create a unique key for this option
        option_key = f"opt_{option.replace(' ', '_')}"

        # Create the HTML for this search result
        is_selected = st.session_state.selected_option == option
        selected_badge = (
            "<span class='selected-badge'>Selected</span>" if is_selected else ""
        )

        # Highlight matching part of the text
        if search_query and search_query.lower() in option.lower():
            start_idx_text = option.lower().find(search_query.lower())
            end_idx_text = start_idx_text + len(search_query)
            highlighted_option = f"{option[:start_idx_text]}<strong>{option[start_idx_text:end_idx_text]}</strong>{option[end_idx_text:]}"
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
                if st.button("Select", key=f"btn_{option}", use_container_width=True):
                    select_option(option)
                    st.session_state.update_page = True
            else:
                st.button(
                    "Selected ✓",
                    key=f"btn_{option}",
                    disabled=True,
                    use_container_width=True,
                )

    # Display pagination controls if there are multiple pages
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            if st.button("◀ Previous", disabled=st.session_state.page_number <= 0):
                st.session_state.page_number -= 1
                st.rerun()

        with col2:
            st.markdown(
                f"<p align='center'>Page {st.session_state.page_number + 1} of {total_pages}</p>",
                unsafe_allow_html=True,
            )

        with col3:
            if st.button(
                "Next ▶", disabled=st.session_state.page_number >= total_pages - 1
            ):
                st.session_state.page_number += 1
                st.rerun()


def get_job_status(job_id: str) -> str:
    """Get the status of a job from the GJP server."""
    try:
        response = requests.get(
            f"http://{GJP_ADDRESS}/db/query",
            params={"q": f"SELECT * FROM jobs WHERE job_id = '{job_id}'"},
        )
        response.raise_for_status()
        result = response_to_dict(response)

        if not result:
            return "pending"

        result = result[0]

        active = result.get("active")
        event = json.loads(result.get("event"))

        if active == "0" and event.get("failed", False):
            return "failed"
        elif active == "1":
            return "running"
        elif active == "0":
            return "completed"
        else:
            return "unknown"
    except Exception as e:
        return "error"


def update_simulation_statuses():
    """Update status for all simulations in history."""
    current_time = datetime.datetime.now()

    for params in st.session_state.simulation_history:
        job_id = params.get("Job ID")
        if not job_id:
            continue

        # Only update if more than 30 seconds have passed since last update
        last_update = st.session_state.last_status_update.get(job_id)
        if last_update and (current_time - last_update).total_seconds() < 30:
            continue

        status = get_job_status(job_id)
        params["Status"] = status
        st.session_state.last_status_update[job_id] = current_time


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
    
    /* Scrollable results container */
    .scrollable-results {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #f0f0f0;
        border-radius: 8px;
        padding: 0 10px;
        margin-top: 10px;
    }
    
    /* Limit height of specific containers */
    [data-testid="stVerticalBlock"] .element-container:has(.search-results-container) {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #f0f0f0;
        border-radius: 8px;
        padding: 0 10px;
        margin-top: 10px;
    }
    
    /* Parameter Summary Styling */
    .parameter-value {
        font-size: 1.2rem;
        font-weight: 500;
        background-color: rgba(25, 135, 84, 0.15);
        border-radius: 5px;
        padding: 6px 10px;
        margin: 3px 0 12px 0;
        display: block;
        color: #4caf50;
    }
    
    .parameter-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #9a9a9a;
        margin-bottom: 4px;
    }
    
    .summary-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(49, 51, 63, 0.1);
        margin-top: 10px;
        border-left: 3px solid #4caf50;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create a search bar that looks more like Google
    st.write("### Search")
    search_query = st.text_input(
        "Search query",
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
        # Reset pagination when search query changes
        st.session_state.page_number = 0

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

        # Display results with pagination (5 items per page)
        display_results(filtered_options, search_query, items_per_page=5)

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
            "Temperature (K)",
            min_value=200,
            max_value=400,
            value=300,
            step=1,
            help="Temperature in Kelvin. Higher temperatures increase molecular motion and energy.",
        )
        st.write(f"Selected temperature: {temperature} K")

        # 3. Friction slider
        friction = st.slider(
            "Friction",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Friction coefficient for the Langevin integrator. Higher values increase damping.",
        )
        st.write(f"Selected friction: {friction}")

    with col2:
        # 4. Pressure slider
        pressure = st.slider(
            "Pressure (atm)",
            min_value=1.0,
            max_value=2.0,
            value=1.5,
            step=0.1,
            help="Pressure in atmospheres. Affects the volume of the simulation box.",
        )
        st.write(f"Selected pressure: {pressure} atm")

        # 5. Forcefield dropdown
        forcefield_options = []
        for field in FORCEFIELD_REGISTRY.values():
            field_instance = field()
            # Add recommended configuration first
            if hasattr(field_instance, "recommended_configuration"):
                forcefield_options.append(
                    field_instance.recommended_configuration["FF"]
                )
            # Then add all other forcefields
            forcefield_options.extend(field_instance.forcefields)

        # Remove duplicates while preserving order
        forcefield_options = list(dict.fromkeys(forcefield_options))

        # Set default to first forcefield if not already selected
        if not st.session_state.selected_forcefield:
            st.session_state.selected_forcefield = forcefield_options[0]

        forcefield = st.selectbox(
            "Select Forcefield",
            options=forcefield_options,
            index=forcefield_options.index(st.session_state.selected_forcefield),
            help="Force field for molecular dynamics simulation. Determines how atoms interact.",
        )
        st.session_state.selected_forcefield = forcefield
        st.write(f"Selected forcefield: {forcefield}")

        # 6. Water dropdown - filtered based on selected forcefield
        water_options = []
        for field in FORCEFIELD_REGISTRY.values():
            field_instance = field()
            if (
                forcefield in field_instance.forcefields
                or forcefield == field_instance.recommended_configuration["FF"]
            ):
                # Add recommended water model first
                if hasattr(field_instance, "recommended_configuration"):
                    water_options.append(
                        field_instance.recommended_configuration["WATER"]
                    )
                # Then add all other water models
                water_options.extend(field_instance.waters)
                break

        # Remove duplicates while preserving order
        water_options = list(dict.fromkeys(water_options))

        # Set default to first water model if not already selected
        if (
            not st.session_state.selected_water
            or st.session_state.selected_water not in water_options
        ):
            st.session_state.selected_water = water_options[0]

        water_model = st.selectbox(
            "Select Water Model",
            options=water_options,
            index=water_options.index(st.session_state.selected_water),
            help="Water model for solvation. Affects how water molecules interact with the protein.",
        )
        st.session_state.selected_water = water_model
        st.write(f"Selected water model: {water_model}")

        # 7. Box shape dropdown
        box_shape_options = ["cube", "dodecahedron", "octahedron"]
        box_shape = st.selectbox("Select Box Shape", box_shape_options)
        st.write(f"Selected box shape: {box_shape}")

        # 8. Update interval input (manual entry instead of slider)
        update_interval_hours = st.number_input(
            "Time to Live (hours)",
            min_value=0.5,
            max_value=24.0,
            value=2.0,
            step=0.5,
            help="How frequently the simulation should update (in hours)",
        )
        # Convert hours to seconds for backend
        update_interval_seconds = int(update_interval_hours * 3600)
        st.write(
            f"Selected update interval: {update_interval_hours} hours ({update_interval_seconds} seconds)"
        )

    # Simulation name input - with the default set to selected_option if available
    simulation_name = st.text_input(
        "Simulation Name",
        value=selected_option if selected_option else "",
        placeholder="Enter a name for this simulation run...",
    )

    # Add a run button
    is_prod = st.checkbox("Production Mode", value=False, help="Run in production mode")

    # Add wallet configuration fields
    st.subheader("Wallet Configuration")

    # Get available wallet names
    wallet_names = get_wallet_names()
    if not wallet_names:
        st.warning("No wallets found in ~/.bittensor/wallets/")
        wallet_name = st.text_input(
            "Wallet Name",
            value=st.session_state.wallet_name,
            placeholder="e.g. folding_testnet",
            key="wallet_name_input",
        )
        wallet_hotkey = st.text_input(
            "Wallet Hotkey",
            value=st.session_state.wallet_hotkey,
            placeholder="e.g. v2",
            key="wallet_hotkey_input",
        )
    else:
        # Wallet name dropdown
        wallet_name = st.selectbox(
            "Wallet Name",
            options=wallet_names,
            index=wallet_names.index(st.session_state.wallet_name)
            if st.session_state.wallet_name in wallet_names
            else 0,
            key="wallet_name_input",
        )

        # Get hotkeys for selected wallet
        hotkeys = get_hotkeys(wallet_name)
        if not hotkeys:
            st.warning(f"No hotkeys found for wallet {wallet_name}")
            wallet_hotkey = st.text_input(
                "Wallet Hotkey",
                value=st.session_state.wallet_hotkey,
                placeholder="e.g. v2",
                key="wallet_hotkey_input",
            )
        else:
            # Hotkey dropdown
            wallet_hotkey = st.selectbox(
                "Wallet Hotkey",
                options=hotkeys,
                index=hotkeys.index(st.session_state.wallet_hotkey)
                if st.session_state.wallet_hotkey in hotkeys
                else 0,
                key="wallet_hotkey_input",
            )

    # Update session state with new values
    st.session_state.wallet_name = wallet_name
    st.session_state.wallet_hotkey = wallet_hotkey

    # Create wallet button
    if st.button("Configure Wallet"):
        create_wallet()

    # Show wallet status if configured
    if st.session_state.wallet:
        st.info(f"Current wallet: {st.session_state.wallet.hotkey.ss58_address}")

    run_simulation: bool = st.button("Run Simulation")

    if run_simulation and is_prod:
        try:
            from folding_api.schemas import FoldingParams

            if not st.session_state.wallet:
                raise ValueError(
                    "Please configure your wallet before running the simulation"
                )

            if not selected_option:
                raise ValueError("Please select a PDB ID first")

            # Show loading state
            with st.spinner("Submitting simulation job..."):
                # Create FoldingParams object
                folding_params = FoldingParams(
                    pdb_id=str(selected_option),  # Ensure it's a string
                    source="rcsb",  # Default to RCSB source
                    ff=forcefield,
                    water=water_model,
                    box=box_shape,
                    temperature=temperature,
                    friction=friction,
                    epsilon=1.0,  # Default epsilon value
                )

                def make_request(
                    address: str, folding_params: FoldingParams
                ) -> requests.Response:
                    try:
                        # Convert params to JSON and encode
                        body_bytes = json.dumps(
                            folding_params.model_dump(), default=str, sort_keys=True
                        ).encode("utf-8")

                        # Generate headers using Epistula
                        epistula = Epistula()
                        headers = epistula.generate_header(
                            st.session_state.wallet.hotkey, body_bytes
                        )

                        # Make the request with timeout
                        response = requests.post(
                            f"http://{address}/organic",
                            data=body_bytes,
                            headers=headers,
                            timeout=30,  # Add timeout
                        )
                        response.raise_for_status()  # Raise exception for bad status codes
                        return response
                    except requests.exceptions.Timeout:
                        raise TimeoutError("Request timed out. Please try again.")
                    except requests.exceptions.RequestException as e:
                        raise ConnectionError(f"Failed to connect to server: {str(e)}")

                # Make the request
                response = make_request(
                    address=API_ADDRESS, folding_params=folding_params
                )

                if response.status_code == 200:
                    job_id = response.json()["job_id"]
                    st.success(f"Job submitted successfully with ID: {job_id}")

                    # Get initial status
                    status = get_job_status(job_id)

                    # Create a dictionary of the current parameters
                    current_params = {
                        "Simulation Name": simulation_name
                        if simulation_name
                        else (
                            selected_option if selected_option else "Unnamed Simulation"
                        ),
                        "Selected Option": selected_option,
                        "Temperature": f"{temperature} K",
                        "Friction": friction,
                        "Pressure": f"{pressure} atm",
                        "Forcefield": forcefield,
                        "Water Model": water_model,
                        "Box Shape": box_shape,
                        "Update Interval": f"{update_interval_hours} hours",
                        "Timestamp": datetime.datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Job ID": job_id,
                        "Status": status,
                    }

                    # Add to history
                    st.session_state.simulation_history.append(current_params)
                else:
                    error_msg = response.text
                    try:
                        error_json = response.json()
                        error_msg = error_json.get("detail", error_msg)
                    except:
                        pass
                    st.error(f"Failed to submit job: {error_msg}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.exception(e)  # Show full traceback for unexpected errors

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
        # st.markdown('<div class="summary-box">', unsafe_allow_html=True)

        st.markdown(
            '<div class="parameter-label">Selected Option</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="parameter-value">{selected_option if selected_option else "None"}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="parameter-label">Temperature</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="parameter-value">{temperature} K</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="parameter-label">Friction</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="parameter-value">{friction}</div>', unsafe_allow_html=True
        )

        st.markdown(
            '<div class="parameter-label">Pressure</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="parameter-value">{pressure} atm</div>', unsafe_allow_html=True
        )

        st.markdown(
            '<div class="parameter-label">Forcefield</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="parameter-value">{forcefield}</div>', unsafe_allow_html=True
        )

        st.markdown(
            '<div class="parameter-label">Water Model</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="parameter-value">{water_model}</div>', unsafe_allow_html=True
        )

        st.markdown(
            '<div class="parameter-label">Box Shape</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="parameter-value">{box_shape}</div>', unsafe_allow_html=True
        )

        st.markdown(
            '<div class="parameter-label">Update Interval</div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="parameter-value">{update_interval_hours} hours</div>',
            unsafe_allow_html=True,
        )

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
        "Box Shape": box_shape,
        "Update Interval": f"{update_interval_hours} hours",
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Only add to history if not in production mode
    if not is_prod:
        st.session_state.simulation_history.append(current_params)
        st.success(f"Simulation '{sim_name}' started with the following parameters:")
        st.json(current_params)

# Horizontal divider before history section
st.markdown("---")

# History Section (Below both parameter selection and summary)
st.subheader("Simulation History")

# Add refresh all button
if st.button("🔄 Refresh All Statuses"):
    update_simulation_statuses()
    # Update the session state to trigger a rerun
    st.session_state.simulation_history = st.session_state.simulation_history.copy()
    st.rerun()

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

                        # Add status display with appropriate color
                        status = params.get("Status", "unknown")
                        status_color = {
                            "running": "green",
                            "completed": "blue",
                            "failed": "red",
                            "pending": "orange",
                            "error": "gray",
                            "unknown": "gray",
                        }.get(status, "gray")
                        st.markdown(
                            f"**Status:** <span style='color: {status_color}'>{status.title()}</span>",
                            unsafe_allow_html=True,
                        )

                    with h_col2:
                        st.markdown("**Pressure:**")
                        st.markdown(f"```{params['Pressure']}```")

                        st.markdown("**Forcefield:**")
                        st.markdown(f"```{params['Forcefield']}```")

                        st.markdown("**Water Model:**")
                        st.markdown(f"```{params['Water Model']}```")

                        st.markdown("**Box Shape:**")
                        st.markdown(f"```{params.get('Box Shape', 'cubic')}```")

                        st.markdown("**Update Interval:**")
                        st.markdown(
                            f"```{params.get('Update Interval', '2.0 hours')}```"
                        )

                        st.markdown("**Job ID:**")
                        st.markdown(f"```{params.get('Job ID', 'N/A')}```")

                    st.caption(f"Run on: {params.get('Timestamp', 'Unknown time')}")
