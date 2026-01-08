
import re
import subprocess
from typing import List
import os
import shutil
from pathlib import Path
import time
from metagpt.actions import Action
from metagpt.schema import Message

from qa_module import AsyncQA_Ori
import config_path
import sys
import glob
from Statistics import global_statistics
import json

class RunnerAction(Action):

    PROMPT_TEMPLATE_allrun: str = """
        Your task is to write linux execution command allrun file to meet the user requirement: {requirement}.
        Note that you only need to focus on the requirements for the CFD simulation task without including any additional analysis or explanation, as these additional analysis or explanations have already been taken into account in the previous input files. You only need to set up the command of main CFD task now.
        The input file list is {file_list}.
        Here is a openfoam allrun file similar to the user requirements:
        {tutorial}
        Please take this file as a reference.
        The possible command list is
        {commands}
        The possible run list is
        {runlists}
        Make sure the written linux execution command are coming from the above two lists.
        According to your task, return ```your_allrun_file_here ``` with NO other texts
        """
    PROMPT_TEMPLATE_review: str = """
        Your task is to review linux execution command ({command}) to meet the user requirement: {requirement}.
        with hlep:
        {command_help}
        make sure the command is correctly used. 
        According to your task, return ```your_allrun_file_here ``` with NO other texts
        """
        
    async def run(self, with_messages:List[Message]=None, **kwargs) -> Message:
        

        # allrun_write = "None"

        # if os.path.exists(allrun_file_path):
        #     print('allrun exist')
        #     with open(allrun_file_path, 'r', encoding='utf-8') as allrun_file:

        #         allrun_write = allrun_file.read()

        # if allrun_write == "None":

        #     requirement = with_messages[0].content
        #     async_qa_allrun = AsyncQA_allrun()
        #     runlists = ['isTest', 'getNumberOfProcessors','getApplication','runApplication','runParallel','compileApplication','cloneCase','cloneMesh']
        #     commands = self.read_commands(config_path.Database_PATH)
        #     file_list = self.read_files(config_path.Case_PATH)

        #     find_tutorial = self.read_tutorial()
        #     #print("find_tutorial:",find_tutorial)
        #     case_name = self.get_case_name(find_tutorial)
        #     #print("case_name:",case_name)
        #     allrun_tutorial = self.get_allrun_tutorial(case_name)
        #     #print("allrun_tutorial:",allrun_tutorial)

        #     prompt_allrun = self.PROMPT_TEMPLATE_allrun.format(
        #         requirement=requirement, 
        #         tutorial = allrun_tutorial,
        #         file_list = file_list, 
        #         commands = commands, 
        #         runlists = runlists)
            
        #     rsp = await async_qa_allrun.ask(prompt_allrun) 
        #     result = rsp["result"]
        #     #doc = rsp["source_documents"]
        #     #print("allrun_source_documents:",doc[0])
        #     #print("allrun:",result)
        #     allrun_write = self.parse_allrun(result)
        #     with open(allrun_file_path, 'w') as outfile:  
        #         outfile.write(allrun_write)
                
        #print('allrun_write:',allrun_write)
        # 需要运行 allrun allrun_postprocessing 与 postprocessing_python.py三个程序，分别对其报错进行处理
        print('global_statistics.Executability = ',global_statistics.Executability)
        if_err_move = 0
        if global_statistics.Executability < 3:

            allrun_file_path = f'{config_path.Case_PATH}/Allrun'
            out_file = os.path.join(config_path.Case_PATH, 'Allrun.out')
            err_file = os.path.join(config_path.Case_PATH, 'Allrun.err')

            self.remove_log_files(config_path.Case_PATH)
            if os.path.exists(err_file):
                os.remove(err_file)
            if os.path.exists(out_file):
                os.remove(out_file)
            self.remove_err_files(config_path.Case_PATH)
            self.remove_pro_files(config_path.Case_PATH)
            if_err_move = 1
            dir_path = config_path.Case_PATH
            initial_files = {}
            for subdir in os.listdir(dir_path):
                subdir_path = os.path.join(dir_path, subdir)
                if os.path.isdir(subdir_path):
                    initial_files[subdir] = set(os.listdir(subdir_path))

            print("initial_files:",initial_files)

            local_path = self.run_command(allrun_file_path, out_file, err_file, config_path.Case_PATH)
            # check endTime and folder time
            error_logs = self.check_foam_errors(local_path)

            print('error_logs:',error_logs)

            # commands_run = self.extract_commands_from_allrun_out(out_file)


            # command = self.compare_commands_with_error_logs(commands_run, error_logs)

            if not error_logs:
                print("No error logs found.")
                result = "None"
                print("完成本次任务所需的总token数为:", global_statistics.total_tokens)

                self.check_endtime_and_folder(config_path.Case_PATH)
                self.save_initial_files(initial_files)
            # elif not command:
            #     print("Error: Could not find the erroneous command.")
            #     #result = commands_run
            #     result = ", ".join(commands_run) 
            # else:
            #     print('command line error:', command[0]['command'])
            #     print('error:', command[0]['error_content'])

            #     error_file_name = f"{config_path.Case_PATH}/{command[0]['command']}.err"
            #     self.save_error_content(command[0]['error_content'], error_file_name)
            #     result = command[0]['command']
            #     self.check_time_and_folder(config_path.Case_PATH)
            #     if "mesh" in result.lower():
            #         global_statistics.Executability = 0
            #     else:
            #         if global_statistics.Executability == 0:
            #             global_statistics.Executability = 1
            #             print("Executability:",global_statistics.Executability)
            #     # delete generated files
            #     if config_path.should_stop == False:
            #         final_files = {}
            #         for subdir in os.listdir(dir_path):
            #             subdir_path = os.path.join(dir_path, subdir)
            #             if os.path.isdir(subdir_path):
            #                 final_files[subdir] = set(os.listdir(subdir_path))

            #         print("final_files:",final_files)

            #         for subdir in final_files:
            #             new_files = final_files[subdir] - initial_files.get(subdir, set())
            #             subdir_path = os.path.join(dir_path, subdir)
            #             for file in new_files:
            #                 file_path = os.path.join(subdir_path, file)
            #                 try:
            #                     if os.path.isfile(file_path):
            #                         os.remove(file_path)
            #                         print(f"Already deleted: {file_path}")
            #                     elif os.path.isdir(file_path):
            #                         os.rmdir(file_path)  
            #                         print(f"Already deleted: {file_path}")
            #                 except Exception as e:
            #                     print(f"Delete:{file_path} occur error: {e}")
                    # if error is found, need to compare the new file list and old file list and convert to the old one
        if global_statistics.Executability == 3 and config_path.tasks>=2:        #跑后处理操作

            allrun_file_path = f'{config_path.Case_PATH}/Allrun_postprocessing'
            if os.path.exists(allrun_file_path): 
                Allrun_postprocessing_context = self.read_allrun_file(allrun_file_path)
                print('Allrun_postprocessing_context:', Allrun_postprocessing_context)
                if Allrun_postprocessing_context == ['None']:
                    os.remove(allrun_file_path)
                    result = "None"
                else:
                    out_file = os.path.join(config_path.Case_PATH, 'Allrun_postprocessing.out')
                    err_file = os.path.join(config_path.Case_PATH, 'Allrun_postprocessing.err')
                    if os.path.exists(err_file):
                        os.remove(err_file)
                    if os.path.exists(out_file):
                        os.remove(out_file)
                    if if_err_move==0:
                        #self.remove_log_files(config_path.Case_PATH)
                        self.remove_err_files(config_path.Case_PATH)
                    self.remove_log_files(config_path.Case_PATH)

                    dir_path = config_path.Case_PATH
                    initial_files = {}
                    for subdir in os.listdir(dir_path):
                        subdir_path = os.path.join(dir_path, subdir)
                        if os.path.isdir(subdir_path):
                            initial_files[subdir] = set(os.listdir(subdir_path))
                    self.run_command(allrun_file_path, out_file, err_file, config_path.Case_PATH)
                    # check endTime and folder time 
                    error_logs1 = self.read_error_logs(err_file)
                    error_logs = self.check_foam_errors(config_path.Case_PATH)
                    #error_log2 = self.read_error_logs(err_file)
                    print('error_logs:',error_logs)
                    
                    print('error_logs1:',error_logs1)

                    #print('error_log2:',error_log2)
                    commands_run = self.extract_commands_from_allrun_out(out_file)

                    command = self.compare_commands_with_error_logs(commands_run, error_logs)

                    if not error_logs1 and not error_logs:
                        print("No error logs found.")
                        #result = ", ".join(commands_run)
                        result = "./Allrun_postprocessing"
                        #self.check_endtime_and_folder(config_path.Case_PATH)
                        global_statistics.Executability = 4
                        
                        # 下面需要找到新生成了哪些文件
                        end_time = self.get_end_time(config_path.Case_PATH)
                        print('endTime',end_time)
                        final_files = {}
                        for subdir in os.listdir(dir_path):
                            subdir_path = os.path.join(dir_path, subdir)
                            if os.path.isdir(subdir_path):
                                final_files[subdir] = set(os.listdir(subdir_path))
                        new_end_file_paths = []
                        all_new_files = []
                        for subdir in final_files:
                            new_files = final_files[subdir] - initial_files.get(subdir, set())
                            subdir_path = os.path.join(dir_path, subdir)
                            if new_files:
                                for file in new_files:
                                    file_path = os.path.join(subdir_path, file)
                                    all_new_files.append(file_path)
                                    
                                if re.match(r'^\d+(\.\d+)?$', subdir) and subdir == end_time:  # 检查目录名是否为数字
                                    
                                    new_end_file_paths.append(file_path)

                        print('new_end_file_paths:',new_end_file_paths)
                        print('all_new_files:',all_new_files)
                        if not all_new_files:
                            # 查看是不是已经存在new_files.json,如果存在说明被运行过
                            new_file_json_path = f"{config_path.Case_PATH}/new_files_data.json"
                            if os.path.exists(new_file_json_path):
                                with open(new_file_json_path, 'r') as file:
                                    data = json.load(file)
                                
                                # Extract the arrays from the JSON content
                                all_new_files = data.get("all_new_files", [])
                                new_end_file_paths = data.get("new_end_file_paths", [])
                            else:
                                print('no new file generated after run postprocessing')
                                # 且没有报错
                                sys.exit()
                        else:
                            pattern = r"VTK/[^/]+\.vtk"
                            matched_vtk_files = [file_path for file_path in all_new_files if re.search(pattern, file_path)]
                            # 检查是否存在匹配的文件
                            if not matched_vtk_files:
                                if not new_end_file_paths:
                                    # 可能是后处理出错了，从而没生成vtk
                                    sys.exit()
                            else:
                                # 保存符合条件的路径到 matched_vtk_files
                                print(f"Matched VTK files: {matched_vtk_files}")
                                new_end_file_paths = matched_vtk_files

                            output_data = {
                                "all_new_files": all_new_files,
                                "new_end_file_paths": new_end_file_paths
                            }

                            # 将数据写入 JSON 文件
                            with open(f"{config_path.Case_PATH}/new_files_data.json", "w") as json_file:
                                json.dump(output_data, json_file, indent=4)

                            print(f"Successfully saved in {config_path.Case_PATH}/new_files_data.json")

                    # elif not command:
                    #     print("Error: Could not find the erroneous command.")
                    #     #result = commands_run
                    #     result = ", ".join(commands_run)
                    else:
                        
                        # print('command line error:', command[0]['command'])
                        # print('error:', command[0]['error_content'])
                        #error_logs1 += '\n'
                        if command:
                            error_logs1 += command[0]['error_content']
                        error_file_name = f"{config_path.Case_PATH}/Postprocessing_error.err"
                        self.save_error_content(error_logs1, error_file_name)
                        # result = command[0]['command']
                        result = "./Allrun_postprocessing"
                        
                        if config_path.should_stop == False:
                            final_files = {}
                            for subdir in os.listdir(dir_path):
                                subdir_path = os.path.join(dir_path, subdir)
                                if os.path.isdir(subdir_path):
                                    final_files[subdir] = set(os.listdir(subdir_path))

                            print("final_files:",final_files)

                            for subdir in final_files:
                                new_files = final_files[subdir] - initial_files.get(subdir, set())
                                subdir_path = os.path.join(dir_path, subdir)
                                for file in new_files:
                                    file_path = os.path.join(subdir_path, file)
                                    try:
                                        if os.path.isfile(file_path):
                                            os.remove(file_path)
                                            print(f"Already deleted: {file_path}")
                                        elif os.path.isdir(file_path):
                                            os.rmdir(file_path)  
                                            print(f"Already deleted: {file_path}")
                                    except Exception as e:
                                        print(f"Delete:{file_path} occur error: {e}")
            else:
                result = 'None'
        if global_statistics.Executability == 4 and config_path.tasks>=2:
            
            #postprocessing_data_path = f"{config_path.Case_PATH}/new_files_data.json"
            # 先看有没有python脚本，没有的话就跳过
            python_postprocessing_path = f'{config_path.Case_PATH}/postprocessing_python.py'
            if os.path.exists(python_postprocessing_path): 
                
                result = 'python3 postprocessing_python.py'
                #读取python_postprocessing_path文件，提取出其中的路径，对比新生成的VTK文件的路径
                #如果两个路径一致，则不修改，否则修改
                if config_path.First_time_of_case:
                    vtk_file_path = self.extract_vtk_path(python_postprocessing_path)
                    print('vtk_file_path in python_script:', vtk_file_path)
                    postprocessing_data_path = f"{config_path.Case_PATH}/new_files_data.json"
                    all_new_files, new_file_paths = self.load_new_files_data(postprocessing_data_path)
                    if new_file_paths:
                        vtk_file_path_true = new_file_paths[0]
                        self.update_vtk_path(python_postprocessing_path, vtk_file_path, vtk_file_path_true)
                    else:
                        print('no vtk path found')
                        sys.exit()
                    config_path.First_time_of_case = False

                out_file = os.path.join(config_path.Case_PATH, 'postprocessing_python.out')
                err_file = os.path.join(config_path.Case_PATH, 'postprocessing_python.err')
                if os.path.exists(err_file):
                    os.remove(err_file)
                if os.path.exists(out_file):
                    os.remove(out_file)

                dir_path = config_path.Case_PATH

                initial_files = set(os.listdir(config_path.Case_PATH))
                self.run_command_python(python_postprocessing_path, out_file, err_file, config_path.Case_PATH)
                final_files = set(os.listdir(config_path.Case_PATH))
                new_end_file_paths = []
                all_new_files = []
                print("Initial files:", initial_files)
                print("Final files:", final_files)
                new_files = final_files - initial_files
                new_end_file_paths = [os.path.join(config_path.Case_PATH, file) 
                for file in new_files 
                if file.lower().endswith((".json", ".png"))]
                all_new_files = [os.path.join(config_path.Case_PATH, file) for file in new_files]

                # 运行后首选看有没有报错，如果有报错需要修改报错，需要查询out_file是否输出自己关心的量，如果没有输出，则需要重新修改
                # 读取postprocessing_python.err, 只要不为空，就

                # 在postprocessing的时候需要有一个cfd_postprocessing的任务来确保修改后的程序不会走偏
                # 需要判断有没有报错
                # 下面需要找到新生成了哪些文件

                # 记录初始文件结构
                # final_files = {}
                # for subdir in os.listdir(config_path.Case_PATH):
                #     subdir_path = os.path.join(config_path.Case_PATH, subdir)
                #     if os.path.isdir(subdir_path):
                #         final_files[subdir] = set(os.listdir(subdir_path))

                error_logs = self.read_postprocessing_python_logs()
                
                if not error_logs:
                    print("No error logs found in python_postprocessing.")

                    global_statistics.Executability = 5
                    
                    if new_end_file_paths:

                        print('new_end_file_paths:', new_end_file_paths)
                        print('all_new_files:', all_new_files)
                        
                        if not all_new_files:
                            sys.exit()
                        else:
                            # 检查是否存在匹配的 .json 或 .png 文件
                            if not new_end_file_paths:
                                print('no json or png found')
                                sys.exit()

                        # 保存数据到 JSON 文件
                        output_data = {
                            "all_new_files": all_new_files,
                            "new_end_file_paths": new_end_file_paths
                        }

                        output_path = f"{config_path.Case_PATH}/new_files_data_python.json"
                        with open(output_path, "w") as json_file:
                            json.dump(output_data, json_file, indent=4)

                        print(f"Successfully saved in {output_path}")
                    else:
                        #
                        print('reading old new_end_file_paths_python')
                        new_file_json_path = f"{config_path.Case_PATH}/new_files_data_python.json"
                        if os.path.exists(new_file_json_path):
                            with open(new_file_json_path, 'r') as file:
                                data = json.load(file)
                            
                            # Extract the arrays from the JSON content
                            all_new_files = data.get("all_new_files", [])
                            new_end_file_paths = data.get("new_end_file_paths", [])
                        else:
                            print('no new file generated after run postprocessing')
                            # 且没有报错
                            sys.exit()

                else:
                    print("error logs found in python_postprocessing.")
                    error_file_name = f"{config_path.Case_PATH}/python_script.err"
                    self.save_error_content(error_logs, error_file_name)

                    for file_path in all_new_files:
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                print(f"Already deleted: {file_path}")
                            elif os.path.isdir(file_path):
                                os.rmdir(file_path)  
                                print(f"Already deleted: {file_path}")
                        except Exception as e:
                            print(f"Delete:{file_path} occur error: {e}")

            else:
                result = 'None'
            # extrat postprocessing
            # 查询在case目录下是否有postProcessing文件夹，如果存在，读取文件夹名
            # 判断文件夹名是否与dependent_var有关，
            # 如果有关，则进入该文件夹，读取该文件夹下的文件名（应该跟时间有关）
            # 判断这些跟时间有关的文件名与dependent_var的相关性，应该取哪一个，或哪一些文件读取
            # 将这些文件作为输入给gpt，让其提取出dependent_var
            # 在OpenFOAM的{case_name}模拟中，想要后处理提取{dependent_var}，请分析如何在controlDict中写后处理程序并在运行后得到的postprocessing中利用python程序自动分析得到{dependent_var}
            # 返回一个修改后的controlDict（在inputwriter写完文件之后，再进行第二次的修改），而后返回能读取postprocessing中的{dependent_var}的python程序
            # 在运行至收敛后，执行该python程序，得到{dependent_var}
            # 如果python 程序执行失败，就迭代几次直至成功

        if global_statistics.Executability == 6 and config_path.tasks>=2:
            result = 'Already_run'
        result = 'None'
        sys.exit(0)
        return result


        
    def read_file_content(self, file_path):
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return None
    def read_commands(self, database_path):

        file_path = f"{database_path}/openfoam_commands.txt"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        
        with open(file_path, 'r') as file:
            commands = [line.strip() for line in file if line.strip()]
    
        return commands
    def read_files(self, base_path):
        file_names = []   
        base_depth = base_path.rstrip(os.sep).count(os.sep) 
        for root, dirs, files in os.walk(base_path):
            current_depth = root.rstrip(os.sep).count(os.sep)
            if current_depth == base_depth + 1: 
                for file in files:
                    file_path = os.path.join(root, file)  

                    try:
                        with open(file_path, 'r') as file_handle:
                            content = file_handle.read() 
                            file_names.append(file)
                    except UnicodeDecodeError:
                        print(f"Skipping file due to encoding error: {file_path}")
                        continue
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
        return file_names

    def get_case_name(self, content):
        match = re.search(r'case name:\s*(.+)', content)
        your_task_folder = match.group(1).strip() if match else 'None'
        return your_task_folder
    
    def get_allrun_tutorial(self,case_name):

        filename = 'openfoam_allrun.txt' 
        file_path = f"{config_path.Database_PATH}/{filename}"
        end_marker = 'input_file_end.```'  
        with open(file_path, 'r') as file:  
            lines = file.readlines()  
        extracted_content = []
        found_keyword = False  
        for line in lines:  
            if found_keyword:  
                if end_marker in line:  
                    break  
                extracted_content.append(line)  
            elif case_name in line:  
                found_keyword = True  
                continue  

        return ''.join(extracted_content)  
    
    def parse_allrun(self, allrun_total):
        print('allrun_total:',allrun_total)

        match = re.search(r'```(?:.*?\n)(.*?)\n```', allrun_total, re.DOTALL)
        allrun_text = match.group(1).strip() if match else 'None'
        return allrun_text
    
    def run_command(self, script_path, out_file, err_file, working_dir):
            #script_path = f'bash -c "source /home/xupan/OpenFoam6/OpenFOAM-6/etc/bashrc && source $WM_PROJECT_DIR/bin/tools/RunFunctions && {script_path}"'
            # result = subprocess.run(script_path, shell=True, cwd=working_dir, capture_output=True, text=True)
            # process = subprocess.Popen(['bash', script_path], cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            #上传文件
        import subprocess
        print("开始上传案例文件...")
        script_path = f'bash -c "scp -i /home/xupan/detonationFoam-main/tutorials/ac0g1wbtzx_tycs2.hpccube.com_RsaKeyExpireTime_2026-01-21_09-36-56.txt -P 65141 -r {config_path.Case_PATH} ac0g1wbtzx@tycs2.hpccube.com:/work/home/ac0g1wbtzx/Xp_agent"'
        result = subprocess.run(script_path, shell=True, cwd=config_path.Case_PATH, capture_output=True, text=True)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        print("Return code:", result.returncode)
        #自动化在超算跑代码
        # --- Step 1: 提交任务 ---
        print("开始提交案例文件...")
        key_path = "/home/xupan/detonationFoam-main/tutorials/ac0g1wbtzx_tycs2.hpccube.com_RsaKeyExpireTime_2026-01-21_09-36-56.txt"
        user_host = "ac0g1wbtzx@tycs2.hpccube.com"
        port = "65141"

        remote_case_dir = f"/work/home/ac0g1wbtzx/Xp_agent/{os.path.basename(config_path.Case_PATH)}"

        remote_cmd = (
            f"cd {remote_case_dir} && mv Allrun Allrun.slurm && sbatch Allrun.slurm"
        )

        ssh_cmd = [
            "ssh",
            "-p", port,
            "-i", key_path,
            "-o", "BatchMode=yes",
            user_host,
            remote_cmd
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        print("Return code:", result.returncode)
        print("完成本次任务所需的总token数为:", global_statistics.total_tokens)
        def get_remote_file_size(user_host, key_path, port, remote_file):
            """获取远程文件大小（字节数）"""
            cmd = [
                "ssh", "-p", port, "-i", key_path,
                user_host, f"stat -c %s {remote_file} 2>/dev/null || echo 0"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            try:
                return int(result.stdout.strip())
            except ValueError:
                return 0
        
        def check_remote_outfile_stable(user_host, remote_dir, key_path, port="22", check_interval=1, stable_time=2):
            """
            检查远程目录下的 .out 文件大小是否稳定
            """
            time.sleep(60)  # 初始等待，确保文件开始生成
            # 获取远程目录下所有 .out 文件
            list_cmd = [
                "ssh", "-p", port, "-i", key_path,
                user_host, f"ls -t {remote_dir}/*.out 2>/dev/null | head -n 1"
            ]
            result = subprocess.run(list_cmd, capture_output=True, text=True)
            out_file = result.stdout.strip()

            if not out_file:
                print("⚠️ 未找到 .out 文件，无法监控。")
                return False

            print(f"📄 检测目标输出文件: {out_file}")

            prev_size = get_remote_file_size(user_host, key_path, port, out_file)
            stable_elapsed = 0

            print("⏳ 正在监控文件变化中...")
            while True:
                time.sleep(check_interval)
                curr_size = get_remote_file_size(user_host, key_path, port, out_file)

                if curr_size == prev_size:
                    stable_elapsed += check_interval
                    print(f"📉 文件大小未变化 {stable_elapsed}s")
                else:
                    print(f"📈 文件增长中... 当前大小 {curr_size} 字节")
                    stable_elapsed = 0

                prev_size = curr_size

                if stable_elapsed >= stable_time:
                    print(f"✅ 文件大小 {stable_time}s 内无变化，认为任务已结束。")
                    return True 
        # --- Step 2: 监控 .out 文件 ---
        ok = check_remote_outfile_stable(user_host, remote_case_dir, key_path, port)
        if not ok:
            print("❌ 未检测到稳定状态，跳过下载。")
            return "None"
        # --- Step 3: 下载整个目录 ---
        local_dir = Path(remote_case_dir).name + "_downloaded"
        os.makedirs(local_dir, exist_ok=True)
        # 先在远程端列出所有非 processor* 的文件和文件夹
        list_cmd = (
            f"ssh -p {port} -i {key_path} {user_host} "
            f"'cd {remote_case_dir} && ls -1 | grep -v ^processor'"
        )
        result = subprocess.run(list_cmd, capture_output=True, text=True, shell=True)
        items = result.stdout.strip().splitlines()
        # 过滤空结果
        if not items:
            print("❌ 远程目录为空或命令执行失败。")
        else:
            print("✅ 将下载以下项目：")
            for item in items:
                print("  -", item)

            # 拼接 scp 命令，逐项下载
            for item in items:
                if item.isdigit():
                    remote_path = f"{user_host}:{remote_case_dir}/{item}"
                    cmd = (
                        f"scp -r -P {port} -i {key_path} "
                        f"'{remote_path}' '{local_dir}/{item}'"
                    )
                    print(f"⬇️ 正在下载: {item}")
                    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    if result.returncode != 0:
                        print(f"❌ 下载失败: {item}")
                        print(result.stderr)
                else:
                    remote_path = f"{user_host}:{remote_case_dir}/{item}"
                    cmd = (
                        f"scp -r -P {port} -i {key_path} "
                        f"'{remote_path}' '{local_dir}'"
                    )
                    print(f"⬇️ 正在下载: {item}")
                    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                    if result.returncode != 0:
                        print(f"❌ 下载失败: {item}")
                        print(result.stderr)
        print("✅ 下载完成。")

        # Step 5: 本地执行 paraFoam
        local_path = Path.cwd() / local_dir
        os.chdir(local_path)
        print("🚀 启动 paraFoam ...")
        subprocess.Popen(["bash", "-c", "paraFoam"])      
        return local_path
    
    def run_command_python(self, script_path, out_file, err_file, working_dir):                       #这段有问题
        with open(out_file, 'w') as out, open(err_file, 'w') as err:
            process = subprocess.Popen(['/mnt/d/Ubantu_run/MetaOpenFOAM/metaopenfoam_env/bin/python', script_path], cwd=working_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            out.write(stdout)
            err.write(stderr)
                        
        return "None"

    def check_foam_errors(self, log_dir):
        error_logs = []

        log_files = [f for f in os.listdir(log_dir) if f.endswith('err')]

        for log_file in log_files:
            log_path = os.path.join(log_dir, log_file)
            with open(log_path, 'r') as file:
                lines = file.readlines()

            print('err_file:',log_file)
            error_indices = None
            for i, line in enumerate(lines):
                if 'error' in line.lower() and 'foam' in line.lower():
                    error_indices = i
                    break

            if error_indices is None:
                continue

            start_index = max(0, error_indices - 30)
            end_index = min(len(lines), error_indices + 60)

            error_content = [line.strip() for line in lines[start_index:end_index]]

            if error_content:
                error_logs.append({
                    'file': log_file,
                    'error_content': "\n".join(error_content)
                })

        return error_logs

    def remove_log_files(self, directory):
        log_files = glob.glob(os.path.join(directory, 'log*'))
        for log_file in log_files:
            os.remove(log_file)

    def extract_commands_from_allrun_out(self, allrun_out_path):
        commands = []
        with open(allrun_out_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if line.startswith('Running '):

                command_part = line.split('Running ')[1]

                command = command_part.split(' on ')[0]
                command_true = command.split()[0]
                commands.append(command_true)
        
        return commands
    def compare_commands_with_error_logs(self, commands_run, error_logs):
        comparison_results = []
        for command in commands_run:
            for error_log in error_logs:
                if command in error_log['file']:
                    comparison_results.append({
                        'command': command,
                        'error_content': error_log['error_content']
                    })
                    break  # Assuming one match per command is enough
        return comparison_results
    
    def save_error_content(self, error_content, error_file_name):
        with open(error_file_name, 'w') as file:
            file.write(error_content)
    def remove_err_files(self, directory):

        err_files = glob.glob(os.path.join(directory, '*.err'))

        for err_file in err_files:
            try:
                os.remove(err_file)
                print(f"Deleted file: {err_file}")
            except OSError as e:
                print(f"Error deleting file {err_file}: {e}")
    def remove_pro_files(self, directory):

        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path) and item.startswith('processor'):
                try:
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item_path}")
                except Exception as e:
                    print(f"Error deleting folder {item_path}: {e}")

    def check_endtime_and_folder(self, address):
        control_dict_path = os.path.join(address, 'system', 'controlDict')
        
        if not os.path.isfile(control_dict_path):
            print("controlDict file not found.")
            return
        
        endtime_value = None

        with open(control_dict_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip().startswith('endTime'):
                    endtime_value = line.split('endTime', 1)[1].strip().strip(';').replace(" ", "")
                    break

        if endtime_value is None:
            print("endTime not found in controlDict.")
            return

        for folder_name in os.listdir(address):
            folder_path = os.path.join(address, folder_name)
            if os.path.isdir(folder_path) and folder_name == endtime_value:
                global_statistics.Executability = 3
                print("Executability: 3")
                #global_statistics.loop = 0
                #config_path.should_stop = True


        #print("not reach endTime")
        return
    def check_time_and_folder(self, address):
        control_dict_path = os.path.join(address, 'system', 'controlDict')
        
        if not os.path.isfile(control_dict_path):
            print("controlDict file not found.")
            return
        
        endtime_value = None

        with open(control_dict_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip().startswith('endTime'):
                    endtime_value = line.split('endTime', 1)[1].strip().strip(';').replace(" ", "")
                    break

        if endtime_value is None:
            print("endTime not found in controlDict.")
            return
        
        for folder_name in os.listdir(address):
            folder_path = os.path.join(address, folder_name)
            if os.path.isdir(folder_path):
                if re.search(r'\b(?!0\b)\d+\b', folder_name):
                    global_statistics.Executability = 2
                    print("Executability: 2")
                
        for folder_name in os.listdir(address):
            folder_path = os.path.join(address, folder_name)
            if os.path.isdir(folder_path) and folder_name == endtime_value:
                global_statistics.Executability = 3
                print("(A)\tExecutability: 3")
                #global_statistics.loop = 0
                #config_path.should_stop = True

        #print("not runnable")
        return
    def read_allrun_file(self, file_path):
            # Read the file and keep only the first 30 lines as a single string
        with open(file_path, "r") as file:
            lines = file.readlines()
        
        return lines
    def is_number(self,s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def count_files_and_lines(self,path):
        result = {}
        total_files = 0
        total_lines = 0
        for subdir in os.listdir(path):
            subdir_path = os.path.join(path, subdir)

            if os.path.isdir(subdir_path) and not (
                (self.is_number(subdir) and float(subdir) != 0) or 
                subdir.startswith("processor") or 
                subdir.startswith("post")
            ):
                file_count = 0
                line_count = 0

                for item in os.listdir(subdir_path):
                    item_path = os.path.join(subdir_path, item)
                    if os.path.isfile(item_path):
                        file_count += 1
                        with open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            line_count += len(lines)

                result[subdir] = {'file_count': file_count, 'line_count': line_count}
                total_files += file_count
                total_lines += line_count
        return result, total_files, total_lines
    
    def display_results(self, results, total_files, total_lines):
        print(f"{'Subdirectory':<40} {'File Count':<15} {'Line Count':<15}")
        print("="*70)
        for subdir, counts in results.items():
            print(f"{subdir:<40} {counts['file_count']:<15} {counts['line_count']:<15}")
        
        print("="*70)
        print(f"{'Total':<40} {total_files:<15} {total_lines:<15}")
        
    def read_error_logs(self, file_path):
        try:
            with open(file_path, 'r') as file:
                error_logs = file.read()
            return error_logs
        except FileNotFoundError:
            return f"Error: The file at {file_path} was not found."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
    # def read_postprocessing_python_logs(self):
    #     """
    #     读取 postprocessing_python.err 和 postprocessing_python.out 文件内容并直接拼接返回
    #     """
    #     error_file_path = f"{config_path.Case_PATH}/postprocessing_python.err"
    #     output_file_path = f"{config_path.Case_PATH}/postprocessing_python.out"

    #     logs = []

    #     # 读取 postprocessing_python.err 文件内容
    #     try:
    #         with open(error_file_path, 'r') as error_file:
    #             error_content = error_file.read()
    #             if error_content.strip():  # 确保不是空文件
    #                 logs.append(error_content)
    #     except FileNotFoundError:
    #         pass
    #     except Exception as e:
    #         pass

    #     # 读取 postprocessing_python.out 文件内容
    #     try:
    #         with open(output_file_path, 'r') as output_file:
    #             output_content = output_file.read()
    #             if output_content.strip():  # 确保不是空文件
    #                 logs.append(output_content)
    #     except FileNotFoundError:
    #         pass
    #     except Exception as e:
    #         pass

    #     # 拼接内容并返回
    #     return "\n".join(logs)
    def read_postprocessing_python_logs(self):
        """
        读取 postprocessing_python.err 和 postprocessing_python.out 文件内容并直接拼接返回。
        如果两个文件都存在但内容为空，返回空字符串。
        """
        error_file_path = f"{config_path.Case_PATH}/postprocessing_python.err"
        output_file_path = f"{config_path.Case_PATH}/postprocessing_python.out"

        logs = []

        # 读取 postprocessing_python.err 文件内容
        try:
            with open(error_file_path, 'r') as error_file:
                error_content = error_file.read().strip()  # 去除空白字符
                if error_content:  # 只有内容非空时才添加
                    logs.append(error_content)
        except FileNotFoundError:
            pass
        except Exception as e:
            pass

        # 读取 postprocessing_python.out 文件内容
        try:
            with open(output_file_path, 'r') as output_file:
                output_content = output_file.read().strip()  # 去除空白字符
                if output_content:  # 只有内容非空时才添加
                    logs.append(output_content)
        except FileNotFoundError:
            pass
        except Exception as e:
            pass

        # 拼接内容并返回，若无内容则返回空字符串
        return "\n".join(logs) if logs else ""

    def get_end_time(self, address):
        control_dict_path = os.path.join(address, 'system', 'controlDict')
        if not os.path.isfile(control_dict_path):
            print("controlDict file not found.")
            return
        
        endtime_value = None

        with open(control_dict_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip().startswith('endTime'):
                    endtime_value = line.split('endTime', 1)[1].strip().strip(';').replace(" ", "")
                    break

        if endtime_value is None:
            print("endTime not found in controlDict.")
            return
        
        return endtime_value


    def save_initial_files(self, initial_files):
        """
        将 initial_files 中指定的文件夹内容保存为 JSON 格式，仅保留 'constant', '0', 'system' 文件夹。

        参数:
            initial_files (dict): 包含文件夹和文件列表的字典。
        """
        # 定义需要保留的文件夹
        valid_folders = {'constant', '0', 'system'}
        
        # 过滤只保留指定的文件夹
        filtered_files = {key: value for key, value in initial_files.items() if key in valid_folders}
        
        # 确定保存路径
        save_path = os.path.join(config_path.Case_PATH, "initial_files.json")
        
        # 确保目录存在
        os.makedirs(config_path.Case_PATH, exist_ok=True)
        
        # 保存为 JSON 文件
        with open(save_path, 'w') as json_file:
            # 将集合转换为列表，因为 JSON 不支持集合
            json.dump({key: list(value) for key, value in filtered_files.items()}, json_file, indent=4)
            print(f"Filtered initial files saved to: {save_path}")
    
    def read_new_files_data(self, file_path):
        """
        Check if the given JSON file exists and read its contents into two arrays.

        Parameters:
            file_path (str): Path to the JSON file.

        Returns:
            all_new_files (list): List of all new file paths.
            new_end_file_paths (list): List of new end file paths.
        """
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # Extract the arrays from the JSON content
            all_new_files = data.get("all_new_files", [])
            new_end_file_paths = data.get("new_end_file_paths", [])
            
            return all_new_files, new_end_file_paths
        else:
            print(f"File {file_path} does not exist.")
            return [], []
    def extract_vtk_path(self, python_postprocessing_path):
        # 打开文件读取内容
        with open(python_postprocessing_path, 'r') as file:
            lines = file.readlines()

        # 查找包含 '.vtk' 的行
        for line in lines:
            if '.vtk' in line:
                # 使用正则表达式提取路径
                match = re.search(r"['\"]([^'\"]+\.vtk)['\"]", line)
                if match:
                    return match.group(1)
        return None  # 如果没有找到路径，返回 None
    
    def load_new_files_data(self, json_file_path):
        """
        读取 JSON 文件并返回 all_new_files 和 new_file_paths 列表.
        
        参数:
            json_file_path (str): JSON 文件路径.
            
        返回:
            tuple: 包含 all_new_files 和 new_file_paths 的元组.
        """
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        print("Loaded JSON data:", data)

        all_new_files = data.get("all_new_files", [])
        new_file_paths = data.get("new_end_file_paths", []) 
        return all_new_files, new_file_paths
    
    def update_vtk_path(self, python_postprocessing_path, vtk_file_path, vtk_file_path_true):
        """
        比较给定的两个路径是否一致，如果不一致，更新文件中的路径为真实路径并保存。
        
        :param python_postprocessing_path: Python 文件的路径
        :param vtk_file_path: 从文件中提取的路径
        :param vtk_file_path_true: 真实的路径
        :return: 无返回值，直接修改文件内容
        """
        # 检查路径是否一致
        if vtk_file_path != vtk_file_path_true:
            print(f"路径不一致，正在更新路径：\n旧路径：{vtk_file_path}\n新路径：{vtk_file_path_true}")

            # 打开文件并读取所有行
            with open(python_postprocessing_path, 'r') as file:
                lines = file.readlines()

            # 遍历所有行，找到包含 vtk_file_path 的行并替换路径
            for i in range(len(lines)):
                if vtk_file_path in lines[i]:
                    # 使用正则表达式替换文件中的路径
                    lines[i] = re.sub(re.escape(vtk_file_path), vtk_file_path_true, lines[i])
                    print(f"路径已更新为：{vtk_file_path_true}")
                    break
            
            # 将更新后的内容写回文件
            with open(python_postprocessing_path, 'w') as file:
                file.writelines(lines)

        else:
            print("路径一致，无需更新。")