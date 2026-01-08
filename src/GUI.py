import streamlit as st
import yaml
import subprocess
from pathlib import Path
import os
import time
import json
from datetime import datetime

# 配置页面
st.set_page_config(
    page_title="爆轰仿真控制台",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主容器 */
    .main {
        padding: 0rem 1rem;
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* 聊天消息样式 */
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
    
    /* 历史记录项 */
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
    
    /* 文件上传区域 */
    .upload-section {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* 日志输出区域 */
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
    
    /* 按钮样式 */
    .stButton>button {
        width: 100%;
        border-radius: 0.375rem;
        font-weight: 500;
    }
    
    /* 标题样式 */
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

# 配置文件路径
ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON_SCRIPT = ROOT_DIR / "src/OptMetaOpenfoam.py"
CONFIG_FILE = ROOT_DIR / "inputs/requirment.yaml"
HISTORY_FILE = ROOT_DIR / "simulation_history.json"

# 初始化会话状态
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

# ==================== 侧边栏：历史记录 ====================
with st.sidebar:
    st.title("🕐 历史记录")
    
    if st.button("➕ 新建仿真", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.current_logs = []
        st.session_state.current_session_id = None
        st.rerun()
    
    st.divider()
    
    # 显示历史记录
    if st.session_state.history_records:
        for idx, record in enumerate(reversed(st.session_state.history_records)):
            session_id = record.get("session_id")
            timestamp = record.get("timestamp", "未知时间")
            requirement = record.get("requirement", "无描述")[:50]
            
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
        st.info("暂无历史记录")

# ==================== 主界面 ====================
st.title("🚀 爆轰仿真智能控制台")

# 显示聊天历史
for message in st.session_state.chat_history:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 您</strong>
            <div style="margin-top: 0.5rem;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant":
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 助手</strong>
            <div style="margin-top: 0.5rem;">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    elif role == "log":
        st.markdown(f"""
        <div class="chat-message log-message">
            <strong>📝 仿真日志</strong>
            <div class="log-container" style="margin-top: 0.5rem;">{content}</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== 文件上传区域 ====================
with st.expander("📎 上传文件（可选）", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("网格文件")
        mesh_file = st.file_uploader(
            "上传 blockMeshDict 文件",
            type=["txt", "dict"],
            key="mesh_upload",
            help="OpenFOAM 网格配置文件"
        )
        if mesh_file:
            mesh_path = ROOT_DIR / "uploads" / mesh_file.name
            mesh_path.parent.mkdir(exist_ok=True)
            with open(mesh_path, "wb") as f:
                f.write(mesh_file.read())
            st.success(f"✅ 已上传: {mesh_file.name}")
    
    with col2:
        st.subheader("机理文件")
        mechanism_file = st.file_uploader(
            "上传 setFieldsDict 文件",
            type=["txt", "dict"],
            key="mechanism_upload",
            help="场初始化配置文件"
        )
        if mechanism_file:
            mech_path = ROOT_DIR / "uploads" / mechanism_file.name
            mech_path.parent.mkdir(exist_ok=True)
            with open(mech_path, "wb") as f:
                f.write(mechanism_file.read())
            st.success(f"✅ 已上传: {mechanism_file.name}")

# ==================== 用户输入区域 ====================
st.divider()

# 输入框
user_input = st.text_area(
    "💬 描述您的仿真需求",
    placeholder="例如：我需要模拟一个二维旋转爆轰发动机，初始压力为 1.5 MPa，温度为 300 K...",
    height=100,
    key="user_input"
)

# 高级设置
with st.expander("⚙️ 高级设置"):
    col1, col2, col3 = st.columns(3)
    with col1:
        max_loop = st.number_input("最大迭代次数", min_value=1, max_value=100, value=10)
    with col2:
        batch_size = st.number_input("批次大小", min_value=1, max_value=50, value=10)
    with col3:
        run_times = st.number_input("运行次数", min_value=1, max_value=10, value=1)

# 提交按钮
col1, col2 = st.columns([3, 1])
with col2:
    submit_button = st.button("🚀 开始仿真", type="primary", use_container_width=True)

# ==================== 处理提交 ====================
if submit_button and user_input and not st.session_state.simulation_running:
    st.session_state.simulation_running = True
    
    # 添加用户消息
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # 准备配置
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
    
    # 保存配置
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    
    # 添加助手响应
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"✅ 收到您的需求！正在准备仿真环境...\n\n**配置摘要：**\n- 最大迭代: {max_loop}\n- 批次大小: {batch_size}\n- 运行次数: {run_times}"
    })
    
    # 创建日志占位符
    log_placeholder = st.empty()
    st.session_state.current_logs = []
    
    # 启动仿真进程
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
    
    # 流式输出日志
    log_lines = []
    for line in iter(process.stdout.readline, ''):
        if line:
            log_lines.append(line.rstrip())
            # 只保留最后50行
            if len(log_lines) > 50:
                log_lines.pop(0)
            
            log_content = "\n".join(log_lines)
            log_placeholder.markdown(f"""
            <div class="chat-message log-message">
                <strong>📝 仿真日志（实时）</strong>
                <div class="log-container" style="margin-top: 0.5rem;">{log_content}</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.1)
    
    process.wait()
    
    # 添加完成消息
    st.session_state.chat_history.append({
        "role": "log",
        "content": "\n".join(log_lines)
    })
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "🎉 仿真完成！结果已保存。"
    })
    
    # 保存到历史记录
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

# ==================== 页面底部信息 ====================
st.divider()
st.caption("💡 提示：点击左侧历史记录可查看之前的仿真 | 🔧 基于 OpenFOAM 的智能仿真系统")