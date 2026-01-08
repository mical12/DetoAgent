import streamlit as st
import yaml
import subprocess
from pathlib import Path
import os
import time
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Detonation Simulation Console",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Chat message style */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #f0f0f0;
        border-left: 4px solid #4CAF50;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #2196F3;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .log-message {
        background-color: #1e1e1e;
        color: #d4d4d4;
        border-left: 4px solid #ff9800;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    /* History item */
    .history-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.375rem;
        cursor: pointer;
        transition: background-color 0.2s;
        border: 1px solid #e0e0e0;
    }
    
    .history-item:hover {
        background-color: #e8e8e8;
    }
    
    .history-item-active {
        background-color: #d0d0d0;
        border-color: #2196F3;
    }
    
    /* File upload section */
    .upload-section {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Log output area */
    .log-container {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
        line-height: 1.5;
    }
    
    /* Button style */
    .stButton>button {
        width: 100%;
        border-radius: 0.375rem;
        font-weight: 500;
    }
    
    /* Title style */
    h1 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    h3 {
        color: #34495e;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Configuration file paths
ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_SCRIPT = ROOT_DIR / "src/OptMetaOpenfoam.py"
CONFIG_FILE = ROOT_DIR / "inputs/requirment.yaml"
HISTORY_FILE = ROOT_DIR / "simulation_history.json"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_logs" not in st.session_state:
    st.session_state.current_logs = []
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "history_records" not in st.session_state:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.history_records = json.load(f)
    else:
        st.session_state.history_records = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# ==================== Sidebar: History ====================
with st.sidebar:
    st.title("🕐 History")
    
    if st.button("➕ New Simulation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.current_logs = []
        st.session_state.current_session_id = None
        st.rerun()
    
    st.divider()
    
    # Display history records
    if st.session_state.history_records:
        for idx, record in enumerate(reversed(st.session_state.history_records)):
            session_id = record.get("session_id")
            timestamp = record.get("timestamp", "Unknown time")
            requirement = record.get("requirement", "No description")[:50]
            
            is_active = session_id == st.session_state.current_session_id
            
            if st.button(
                f"📋 {timestamp}\n{requirement}...",
                key=f"history_{idx}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_session_id = session_id
                st.session_state.chat_history = record.get("chat_history", [])
                st.rerun()
    else:
        st.info("No history records")

# ==================== Main Interface ====================
st.title("🚀 Detonation Simulation Control Console")

# Display chat history
for message in st.session_state.chat_history:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You</strong>
            <div style="margin-top: 0.5rem;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant</strong>
            <div style="margin-top: 0.5rem;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "log":
        st.markdown(f"""
        <div class="chat-message log-message">
            <strong>📝 Simulation Log</strong>
            <div class="log-container" style="margin-top: 0.5rem;">{content}</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== File Upload Area ====================
with st.expander("📎 Upload Files (Optional)", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mesh File")
        mesh_file = st.file_uploader(
            "Upload blockMeshDict file",
            type=["txt", "dict"],
            key="mesh_upload",
            help="OpenFOAM mesh configuration file"
        )
        if mesh_file:
            mesh_path = ROOT_DIR / "uploads" / mesh_file.name
            mesh_path.parent.mkdir(exist_ok=True)
            with open(mesh_path, "wb") as f:
                f.write(mesh_file.read())
            st.success(f"✅ Uploaded: {mesh_file.name}")
    
    with col2:
        st.subheader("Mechanism File")
        mechanism_file = st.file_uploader(
            "Upload setFieldsDict file",
            type=["txt", "dict"],
            key="mechanism_upload",
            help="Field initialization configuration file"
        )
        if mechanism_file:
            mech_path = ROOT_DIR / "uploads" / mechanism_file.name
            mech_path.parent.mkdir(exist_ok=True)
            with open(mech_path, "wb") as f:
                f.write(mechanism_file.read())
            st.success(f"✅ Uploaded: {mechanism_file.name}")

# ==================== User Input Area ====================
st.divider()

# Input box
user_input = st.text_area(
    "💬 Describe Your Simulation Requirements",
    placeholder="Example: I need to simulate a 2D rotating detonation engine with initial pressure of 1.5 MPa and temperature of 300 K...",
    height=100,
    key="user_input"
)

# Advanced settings
with st.expander("⚙️ Advanced Settings"):
    col1, col2, col3 = st.columns(3)
    with col1:
        max_loop = st.number_input("Max Iterations", min_value=1, max_value=100, value=10)
    with col2:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=50, value=10)
    with col3:
        run_times = st.number_input("Run Times", min_value=1, max_value=10, value=1)

# Submit button
col1, col2 = st.columns([3, 1])
with col2:
    submit_button = st.button("🚀 Start Simulation", type="primary", use_container_width=True)

# ==================== Handle Submission ====================
if submit_button and user_input and not st.session_state.simulation_running:
    st.session_state.simulation_running = True
    
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Prepare configuration
    config = {
        "usr_requirment": user_input,
        "BLOCKMESHDICT_PATH": str(ROOT_DIR / "uploads" / mesh_file.name) if mesh_file else "",
        "SETFIELDS_PATH": str(ROOT_DIR / "uploads" / mechanism_file.name) if mechanism_file else "",
        "max_loop": max_loop,
        "batchsize": batch_size,
        "run_times": run_times,
        "temperature": 0.0,
        "searchdocs": 2,
        "MetaGPT_PATH": "/mnt/d/Ubantu_run/MetaOpenFOAM/MetaGPT",
        "DEEPSEEK_API_KEY": "5fca652c-6dac-49a1-9cdb-cf50de781d13",
        "DEEPSEEK_BASE_URL": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "model": "deepseek-r1-250120"
    }
    
    # Save configuration
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    
    # Add assistant response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"✅ Requirements received! Preparing simulation environment...\n\n**Configuration Summary:**\n- Max Iterations: {max_loop}\n- Batch Size: {batch_size}\n- Run Times: {run_times}"
    })
    
    # Create log placeholder
    log_placeholder = st.empty()
    st.session_state.current_logs = []
    
    # Start simulation process
    env = os.environ.copy()
    env["CONFIG_FILE_PATH"] = str(CONFIG_FILE)
    
    process = subprocess.Popen(
        ["python", str(PYTHON_SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=ROOT_DIR,
        text=True,
        bufsize=1
    )
    
    # Stream log output
    log_lines = []
    for line in iter(process.stdout.readline, ''):
        if line:
            log_lines.append(line.rstrip())
            # Keep only the last 50 lines
            if len(log_lines) > 50:
                log_lines.pop(0)
            
            log_content = "\n".join(log_lines)
            log_placeholder.markdown(f"""
            <div class="chat-message log-message">
                <strong>📝 Simulation Log (Real-time)</strong>
                <div class="log-container" style="margin-top: 0.5rem;">{log_content}</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.1)
    
    process.wait()
    
    # Add completion message
    st.session_state.chat_history.append({
        "role": "log",
        "content": "\n".join(log_lines)
    })
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "🎉 Simulation completed! Results have been saved."
    })
    
    # Save to history
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_session_id = session_id
    
    history_entry = {
        "session_id": session_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requirement": user_input,
        "chat_history": st.session_state.chat_history
    }
    
    st.session_state.history_records.append(history_entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.history_records, f, ensure_ascii=False, indent=2)
    
    st.session_state.simulation_running = False
    st.rerun()

# ==================== Page Footer ====================
st.divider()
st.caption("💡 Tip: Click on history records in the sidebar to view previous simulations | 🔧 Intelligent simulation system based on OpenFOAM")