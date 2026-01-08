import subprocess
from pathlib import Path

def save_postprocess_commands(file_path):
    try:
        # 执行 postProcess -list 命令
        result = subprocess.run(["postProcess", "-list"], stdout=subprocess.PIPE, text=True)
        command_list = result.stdout.strip().split("\n")
        
        # 找到包含 functionObjects 的那一行及其后续内容
        function_objects_index = None
        for i, line in enumerate(command_list):
            if "functionObjects" in line:
                function_objects_index = i
                break
        
        if function_objects_index is not None:
            # 从包含 functionObjects 的行开始保存
            filtered_command_list = command_list[function_objects_index:]
            
            # 将命令列表保存到指定路径的文件中
            with open(Path(file_path) / "postprocessing_commands.txt", "w") as file:
                for command in filtered_command_list:
                    file.write(command + "\n")
            
            print(f"Postprocessing commands saved to {file_path}/postprocessing_commands.txt")
        else:
            print("No 'functionObjects' found in the postProcess command list.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error while executing postProcess command: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 读取保存的 postProcess 命令列表
def read_postprocess_commands(file_path):
    try:
        with open(Path(file_path) / "postprocessing_commands.txt", "r") as file:
            command_list = file.read().splitlines()
        return command_list
    except FileNotFoundError:
        print(f"File {file_path}/postprocessing_commands.txt not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# 示例路径

file_path = '/data/Chenyx/OptMetaOpenFOAM/database'

# 保存 postProcess 命令列表
save_postprocess_commands(file_path)

# 读取 postProcess 命令列表
commands = read_postprocess_commands(file_path)
if commands:
    print("Postprocessing Commands List:")
    for command in commands:
        print(command)
