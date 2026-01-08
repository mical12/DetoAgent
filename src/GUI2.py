import streamlit as st
import yaml
import subprocess
from pathlib import Path
import os
import time
import sys
sys.path.append("/mnt/d/Ubantu_run/MetaOpenFOAM")

# 配置文件路径 
ROOT_DIR = Path(__file__).resolve().parent.parent  # 项目根目录
PYTHON_SCRIPT = ROOT_DIR / "src/OptMetaOpenfoam.py"
CONFIG_FILE = ROOT_DIR / "inputs/requirment.yaml"
st.title("🚀 自动化爆轰仿真控制界面")

# 1️⃣ 读取当前配置
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {"usr_requirment": None, "BLOCKMESHDICT_PATH":"","SETFIELDS_PATH":"","max_loop":10,"temperature":0.0,"batchsize":10,"searchdocs":2,"run_times":1,"MetaGPT_PATH":"/mnt/d/Ubantu_run/MetaOpenFOAM/MetaGPT","DEEPSEEK_API_KEY":"5fca652c-6dac-49a1-9cdb-cf50de781d13","DEEPSEEK_BASE_URL":"https://ark.cn-beijing.volces.com/api/v3/chat/completions","model":"deepseek-r1-250120"}

# 2️⃣ 动态展示参数输入
st.subheader("仿真参数设置")

usr_requirment = st.text_input("您的爆轰仿真需求", value=config.get("usr_requirment",""))
BLOCKMESHDICT_PATH = st.text_input("您的自定义网格文件", value=config.get("BLOCKMESHDICT_PATH",""))
SETFIELDS_PATH = st.text_input("您的自定义初始化场文件", value=config.get("SETFIELDS_PATH",""))

# 3️⃣ 更新配置
config.update({
    "usr_requirment": usr_requirment,
    "BLOCKMESHDICT_PATH": BLOCKMESHDICT_PATH,
    "SETFIELDS_PATH": SETFIELDS_PATH
})

# 4️⃣ 保存并运行
if st.button("💥 运行仿真"):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    
    st.write("✅ 配置已保存:", CONFIG_FILE)
    st.info("正在启动仿真进程...")

    # 注意：通过subprocess传入环境变量 CONFIG_FILE_PATH
    # 环境变量：在现有系统环境的基础上添加 CONFIG_FILE_PATH
    env = os.environ.copy()
    env["CONFIG_FILE_PATH"] = str(CONFIG_FILE)
    process = subprocess.Popen(
        ["python", str(PYTHON_SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=ROOT_DIR,  # 关键：切换到项目根目录
        text=True
    )

    
    # ⭐ 本地日志文件路径
    LOG_FILE = ROOT_DIR / "simulation.log"
    log_f = open(LOG_FILE, "a", encoding="utf-8")
    st.write(f"📝 仿真日志将保存到: {LOG_FILE}")

    st.subheader("仿真日志输出：")
    log_area = st.empty()
    lines = []
    last_display = ""  # 保存上一次显示内容
    MAX_LINES = 20  # 最多显示 20 行
    for line in iter(process.stdout.readline, ''):
        # ⭐ 写入本地日志
        log_f.write(line)
        log_f.flush()
        lines.append(line.rstrip())
        display_lines = lines[-MAX_LINES:]
        current_display = "\n".join(display_lines)

        # 只有和上一次显示内容不同才更新
        if current_display != last_display:
            log_area.text(current_display)
            last_display = current_display
            time.sleep(0.24)  # 控制刷新频率

    process.wait()
    log_f.close()  # ⭐ 关闭日志文件
    st.success("仿真完成！")

