
import re
from typing import List
import os
from metagpt.actions import Action
from metagpt.schema import Message
from qa_module import AsyncQA_tutorial, AsyncQA_Ori, AsyncQA_allrun
import config_path
import subprocess
import sys
import json
from pathlib import Path
from Statistics import global_statistics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml


class InputWriterAction(Action):

    PROMPT_TEMPLATE: str = """
    假如你是一名爆轰专家，你现在的任务是利用openfoam6编程，完成{requirement}.
    以下是高度类似的，已经跑通过的案例的foamfile:
    {tutorial_file}
    这个foamfile文件的参数设置是正确的，一般而言无需修改内容，如果用户有明确需求需要修改某个参数才进行更改.
    根据你的任务, 返回 ```你的代码 ``` ，不要有其他文字。
    你的代码:
    """

    metaopenfoam_PROMPT_TEMPLATE_allrun: str = """
        Your task is to write linux execution command allrun file to meet the user requirement: {requirement}.
        Note that you only need to focus on the requirements for the CFD simulation task without including any additional analysis or explanation (like postprocessing), as these additional analysis or explanations have already been taken into account in the previous input files. You only need to set up the command of main CFD task now (like generate grids, run preprocessing, and run XXfoam).
        The input file list is {file_list}.
        Here is a openfoam allrun file similar to the user requirements:
        {tutorial}
        Please take this file as a reference.
        The possible command list is
        {commands}
        In the command list, the following commands are **forbidden** and should **never** be used:
            - `setFields`
            - `changeDictionary`
        The possible run list is
        {runlists}
        Make sure the written linux execution command are coming from the above two lists.
        According to your task, return 
        ```bash
        your_allrun_file_here
        ``` 
        with **no other texts**. And replace the placeholder your_allrun_file_here with the actual Allrun file content. Do not return the placeholder, but instead return the actual file content.
        """

    METAOPENFOAM_PROMPT_TEMPLATE: str = """
    Your task is {requirement}.
    The similar foamfile is provided as follows:
    {tutorial_file}
    Please take this foamfile as a reference, which may help you to finish your task.
    According to your task, return ```your_code_here ``` with NO other texts,
    your code:
    """
    SETFIELDS_PROMPT_TEMPLATE: str = """
    假如你是一名爆轰专家，你现在的任务是利用openfoam6编程，完成{requirement}.
    以下是高度类似的，已经跑通过的案例的foamfile:
    {tutorial_file}
    这个foamfile文件的参数设置是正确的，还有供参考的已经完成的网格文件信息{blockdict}.还有供参考的已经完成的0文件夹下所有文件信息{tutorial_zero_files}.只允许修改或复制参考案例中已经存在的字段与结构，不补充任何用户未给出的数值。严禁：
   - 假设任何物种比例、温度、压力、速度数值，setFieldsDict 仅用于局部初始化，不用于完整初始条件定义.
    根据你的任务, 返回 ```你的代码 ``` ，不要有其他文字。
    你的代码:
    """

    SETFIELDS_PROMPT_thermophysicalProperties: str = """
    假如你是一名爆轰专家，你现在的任务是利用openfoam6编程，完成{requirement}.
    以下是高度类似的，已经跑通过的案例的foamfile:
    {tutorial_file}
    这个foamfile文件的参数设置是正确的，且在该文件夹中存在{chemical_mechanism_file}机理文件可以引用，一般而言无需修改内容，如果用户有明确需求需要修改某个参数才进行更改.
    根据你的任务, 返回 ```你的代码 ``` ，不要有其他文字。
    你的代码:
    """

    PROMPT_blockMeshDict_Modify_TEMPLATE: str = """
    假如你是一名爆轰专家，你现在的任务是利用openfoam6编程，完成 {requirement}.
    之前已经写完一份blockDict文件:<{blockDict}>，并用blockMesh命令发现问题如下:
    <{Questions}>
    请将以上信息作为参考这可能有助于你完成任务。修改blockDict文件代码。
    根据你的任务，返回：```你的代码 ``` （不要添加任何其他文字），
    你的代码：
    """

    PROMPT_blockMeshDict_TEMPLATE: str = """
    假如你是一名爆轰专家，你现在的任务是利用openfoam6编程，完成 {requirement}.
    以下是类似的，已经跑通过的案例的foamfile:
    {tutorial_file}
    请参考这个foamfile文件，看是否满足任务需求，如果满足就不用修改，如果不满足请以符合openfoam语法规范的形式进行修改.
    根据你的任务, 返回 ```你的代码 ``` ，不要有其他文字。
    你的代码:
    """
    

    PROMPT_Initial_pt_physical_quantity_TEMPLATE: str = """
    假如你是一名爆轰专家，你现在的任务是利用openfoam6编程，完成 {requirement}.当量比，指的混合气体中摩尔量之比，在openfoam中初始化气体物理量时填入文件中的是质量分数，Ydafault文件中所有值为0
    之前已经完成的 blockMeshDict 文件如下:
    <{blockMeshDict_file}>
    以下是类似的，已经跑通过的案例的foamfile:
    <{tutorial_file}>
    请将以上信息作为参考这可能有助于你完成任务，当你需要nonuniform List<scalar>，你只需编写网格数以及();。();内不要加数值、省略号等字符串，每个cell的值会由后续脚本填充。
    根据你的任务，返回：```你的代码 ``` （不要添加任何其他文字），
    你的代码：
    """
    PROMPT_Initial_physical_quantity_TEMPLATE: str = """
    假如你是一名爆轰专家，你现在的任务是利用openfoam6编程，完成 {requirement}.当量比，指的混合气体中摩尔量之比，在openfoam中初始化气体物理量时填入文件中的是质量分数，Ydafault文件仅作为运行所需文件，不需要初始化各气体质量分数，设置为0即可。
    之前已经完成的 blockMeshDict 文件如下:
    <{blockMeshDict_file}>
    以下是类似的，已经跑通过的案例的foamfile:
    <{tutorial_file}>
    一般而言无需修改内容，如果用户有明确需求需要修改某个参数才进行更改。
    根据你的任务，返回：```你的代码 ``` （不要添加任何其他文字），
    你的代码：
    """

    PROMPT_topsetdict_TEMPLATE: str = """
    假如你是一名爆轰专家，你现在的任务是利用openfoam6编程，完成 {requirement}.当量比，指的混合气体中摩尔量之比，在openfoam中初始化气体物理量时填入文件中的是质量分数，Ydafault文件仅作为运行所需文件，不需要初始化各气体质量分数，设置为0即可。
    之前已经完成的 controlDict_content 文件如下:
    <{controlDict_content}>
    以下是类似的，已经跑通过的案例的foamfile:
    <{tutorial_file}>
    一般而言无需修改内容，如果用户有明确需求需要修改某个参数才进行更改。
    根据你的任务，返回：```你的代码 ``` （不要添加任何其他文字），
    你的代码：
    """
     
    PROMPT_Initial_physical_quantity_TEMPLATE2: str = """
    假如你是一名爆轰专家，你现在的任务是利用openfoam6编程，完成 {requirement}.
    之前修改过的文件如下:
    <{file}>
    以下是类似的，已经跑通过的案例的foamfile:
    <{tutorial_file}>
    请将以上信息作为参考这可能有助于你完成任务。现在只需要在修改过的文件的nonuniform List<scalar>下填充（）内的值即可。我已经构造了一个填充脚本，现在你只需要提供以下字段：
        {{total_cells: 总单元格数量
        high_pressure_cells: 前多少个单元为高压
        high_pressure: 高压值 (Pa)
        low_pressure: 低压值 (Pa)}}
    根据你的任务，返回以上字典形式结果 （不要添加任何其他文字），
    你的回复：
    """







    PROMPT_select_physical_quantity_TEMPLATE: str = """
    user request is {requirement}.
    The previously completed file momentumTransport is shown below:
    <{momentumTransport_file}>
    The similar foamfile is provided as follows:
    <{tutorial_file}>
    Please take these foamfiles as a reference, which may help you to finish your task.
    Your task is to determine which files need to be written in the 0 folder to meet the user requirements.
    Please list the names of the files that need to be written, and return them in list format.
    The file names should be in the format of 'file_name', without any additional text.
    """

    PROMPT_TEMPLATE_no_tutorial: str = """
    Your task is {requirement}.
    According to your task, return ```your_code_here ``` with NO other texts,
    your code:
    """
    PROMPT_Find: str = """
        Find the OpenFOAM foamfile that most closely matches the following foamfile:
        {file_name} in {file_folder} of case name: {case_name}
    """
    name: str = "InputWriterAction"

    PROMPT_TEMPLATE_allrun: str = """
        Your task is to write linux execution command allrun file to meet the user requirement: {requirement}.
        Note that you only need to focus on the requirements for the CFD simulation task without including any additional analysis or explanation (like postprocessing), as these additional analysis or explanations have already been taken into account in the previous input files. You only need to set up the command of main CFD task now (like generate grids, run preprocessing, and run XXfoam).
        The input file list is {file_list}.
        Here is a openfoam allrun file similar to the user requirements:
        {tutorial}
        Please take this file as a reference.

        In addition, a previously generated decomposeParDict file is used to specify the required total number of processor cores.:
        {decomposeParDict}
        Here is a completed openfoam controlDict file,please refer to the solvers used in it.:
        {controlDict}
        The possible command list is
        {commands}
        In the command list, the following commands are **forbidden** and should **never** be used:
            - `setFields`
            - `changeDictionary`
        The possible run list is
        {runlists}
        Make sure the written linux execution command are coming from the above two lists.
        According to your task, return 
        ```bash
        your_allrun_file_here
        ``` 
        with **no other texts**. And replace the placeholder your_allrun_file_here with the actual Allrun file content. Do not return the placeholder, but instead return the actual file content.
        """
    PROMPT_TEMPLATE_allrun_rewrite: str = """
    Your task is {requirement}.
    The similar foamfile is provided as follows:
    {tutorial}
    Please take this foamfile as a reference, which may help you to finish your task.
    And please do not use changeDictionary command.
    According to your task, return 
    ```bash
    your_allrun_file_here
    ``` 
    with **no other texts**. And replace the placeholder your_allrun_file_here with the actual Allrun file content. Do not return the placeholder, but instead return the actual file content.
    """
    PROMPT_TEMPLATE_postprocessing_total: str = """
In the OpenFOAM simulation for '{CFD_task}', to post-process and extract '{dependent_var}', please analyze how to write the post-processing command in linux Allrun_postprocessing file and use a Python script to automatically analyze the '{dependent_var}' from the results generated in the postprocessing stage after the simulation runs.
note that the postprocessing function can only be selected from the following list:
{postprocessing_list}
Please first return the 'Allrun_postprocessing' file for linux execution as:
`Allrun_postprocessing` file begin ```
your_Allrun_postprocessing_here 
``` `Allrun_postprocessing` file end
with **no other texts**. And replace the placeholder your_Allrun_postprocessing_here with the actual Allrun_postprocessing file content. Do not return the placeholder, but instead return the actual file content.
And then return the corresponding Python script as:
Python script begin ```
your_python_code_here
``` Python script end
with **no other texts**. And replace the placeholder your_python_code_here with the actual generated Python script. Do not return the placeholder, but instead return the actual file content.
        """
    PROMPT_TEMPLATE_postprocessing_allrun: str = """
In the OpenFOAM simulation for '{CFD_task}', to post-process and extract '{dependent_var}', please first write the post-processing command to get the openfoam post-processing file, which could be used for a Python script to extract '{dependent_var}'
note that the postprocessing function can only be selected from the following list:
{postprocessing_list}
Please first return the 'Allrun_postprocessing' file for linux execution as:
`Allrun_postprocessing` file begin ```
your_Allrun_postprocessing_here
``` `Allrun_postprocessing` file end
with **no other texts**. And replace the placeholder your_Allrun_postprocessing_here with the actual Allrun_postprocessing file content. Do not return the placeholder, but instead return the actual file content.
        """
    PROMPT_TEMPLATE_postprocessing_allrun2: str = """
In the OpenFOAM simulation for '{CFD_task}', to post-process and extract '{dependent_var}', please first write the post-processing command to get the openfoam post-processing file, which could be used for a Python script to extract '{dependent_var}'
Note that the previous allrun for CFD task has already been executed, so you only need to provide the post-processing command. Do not include 'runApplication blockMesh' or any 'runApplication &Application' commands.
Additionally, do not include any if, echo, exit, or other error handling commands. 
The post-processing function can only be selected from the following list:
{postprocessing_list}
You can call the above post-processing function using either 'runApplication postProcess -func Specific_postprocessing_function' or '&Application -postProcess -func Specific_postprocessing_function'. 
Note that the former invokes postProcess for post-processing, while the latter uses the solver for post-processing. The choice between the two depends on the type of Specific_postprocessing_function.
Please source the tutorial run functions first by including this line:
. $WM_PROJECT_DIR/bin/tools/RunFunctions
Please first return the 'Allrun_postprocessing' file for linux execution as:
```Allrun_postprocessing
your_Allrun_postprocessing_here
```
with **no other texts**. And replace the placeholder your_Allrun_postprocessing_here with the actual Allrun_postprocessing file content. Do not return the placeholder, but instead return the actual file content.
        """
    PROMPT_TEMPLATE_postprocessing_allrun3: str = """
In the OpenFOAM simulation for '{CFD_task}', the CFD postprocessing task is '{CFD_postprocessing_task}'
please first write the post-processing command to get the openfoam post-processing file, which could be used for a Python script to complete the post-processing task.
And then transform the generated postprocessing file into VTK format by including this line:
foamToVTK -latestTime -fields '(Specific_postprocessing_file1 Specific_postprocessing_file2 ...)' ...
Note that the previous allrun for CFD task has already been executed, so you only need to provide the post-processing command. Do not include 'runApplication blockMesh' or any 'runApplication &Application' commands.
Additionally, do not include any if, echo, exit, or other error handling commands. 
The post-processing function can only be selected from the following list:
{postprocessing_list}
You can call the above post-processing function using either 'runApplication postProcess -func Specific_postprocessing_function' or '&Application -postProcess -func Specific_postprocessing_function'. 
Note that the former invokes postProcess for post-processing, while the latter uses the solver for post-processing. The choice between the two depends on the type of Specific_postprocessing_function.
Please source the tutorial run functions first by including this line:
. $WM_PROJECT_DIR/bin/tools/RunFunctions

Please return the 'Allrun_postprocessing' file for linux execution as:
```Allrun_postprocessing
your_Allrun_postprocessing_here
```
with **no other texts**. And replace the placeholder your_Allrun_postprocessing_here with the actual Allrun_postprocessing file content. Do not return the placeholder, but instead return the actual file content.
        
        """
    PROMPT_TEMPLATE_postprocessing_allrun_vtk: str = """
In the OpenFOAM simulation for '{CFD_task}', the CFD postprocessing task is '{CFD_postprocessing_task}'.
The required file list for post-processing is '{related_file_list}'.
please write the post-processing command to transform the generated postprocessing file into VTK format by including this line:
foamToVTK -latestTime -fields '(Specific_postprocessing_file1 Specific_postprocessing_file2 ...)' ...
Please source the tutorial run functions first by including this line:
. $WM_PROJECT_DIR/bin/tools/RunFunctions
Please return the 'Allrun_postprocessing' file for linux execution as:
```Allrun_postprocessing
your_Allrun_postprocessing_here
```
with **no other texts**. And replace the placeholder your_Allrun_postprocessing_here with the actual Allrun_postprocessing file content. Do not return the placeholder, but instead return the actual file content.
        
        """
    
    PROMPT_TEMPLATE_postprocessing_if_exist: str = """
In the OpenFOAM simulation for '{CFD_task}', the CFD postprocessing task is '{CFD_postprocessing_task}'.  
Before performing the postprocessing task, check whether the required files are present in the list '{file_list}'.  

For example:  
- If the postprocessing task involves `yPlus`, the list should contain the file `yPlus`.  
- If the postprocessing task involves velocity fields, the list should contain the file `U`.  

If the required file is present, return:  
{json_structure}
Otherwise, return:  
```No```
    """

    PROMPT_TEMPLATE_postprocessing_allrun_JSON: str = """
In the OpenFOAM simulation for '{CFD_task}', to post-process and extract '{dependent_var}', please first write the post-processing command to get the openfoam post-processing file, which could be used for a Python script to extract '{dependent_var}'
note that the postprocessing function can only be selected from the following list:
{postprocessing_list}
Please first return the 'Allrun_postprocessing' file for linux execution in the following JSON format:
```
{JSON_allrun_postprocessing}
```
with **no other texts**.
        """
        
    PROMPT_TEMPLATE_postprocessing_python: str = """
In the OpenFOAM simulation for '{CFD_task}', to post-process and extract '{dependent_var}', the following Linux command was executed:
```
{postprocessing_command}
```
This command generated a file at the following path: '{postprocessing_new_data_path}'.
Here are the first 50 lines of the file (if the file contains more than 50 lines):
```
{postprocessing_data}
```
Please write a Python script that reads this file, extracts '{dependent_var}', and saves it in the following JSON format as 'dependent_var.json' in the current directory of the Python script: 
```
{JSON_dependent_var}
```
Please return the corresponding Python script as:
```python
your_python_code_here
```
with **no other texts**. And replace the placeholder your_python_code_here with the actual generated Python script. Do not return the placeholder, but instead return the actual file content.
        """
    
    PROMPT_TEMPLATE_postprocessing_python_for_vtk: str = """

In the OpenFOAM simulation for '{CFD_task}', the CFD postprocessing task is '{CFD_postprocessing_task}'.  
The following Linux command was executed:  
```  
{postprocessing_command}  
```  
This command generated a VTK file at the following path: '{postprocessing_new_data_path}'.  
Please write a Python script that reads this file and completes the CFD postprocessing task: 
- If the CFD postprocessing task involves extracting a specific value, extract it and save it in the following JSON format as 'postprocessing_var.json' in the script's current directory:  
```  
{JSON_dependent_var}  
```  
- If the CFD postprocessing task involves plotting, generate the required plot and save it as a PNG file in the script's current directory.  

Please return the Python script in the following format:  
```python  
your_python_code_here  
```  
Replace the placeholder `your_python_code_here` with the actual Python script code. Do not include any other text, and provide only the complete Python script as the output.
        """
    PROMPT_TEMPLATE_python_env: str = """
For the following Python program:  
{python_text}  
Please return the required packages in the following format:
Python env list begin ```
Your_python_env_list_here
``` Python env list end
with **no other texts**.
        """
    PROMPT_TEMPLATE_postprocessing_rewrite: str = """
When a Python program encounters an error during execution:  
{error}  
Here is the Python script being executed, 
{python_text}  

Please determine whether this error is caused by the Python environment or the content of the program.  
If the error is due to the Python environment, return a Linux command using Python subprocess to install or update the required library version as:  
###  
python_linux_command  
###  
with **no other texts**. 

If the error is due to the program content, return the corrected Python script as:  
```
Python_code_here  
```  
with **no other texts**.  
        """
    async def run(self, with_messages:List[Message]=None, **kwargs) -> Message:
        def generate_internal_field(old_code_text, total_cells=1000, high_pressure_cells=10,
                            high_pressure=9119250, low_pressure=101325):
            """
            生成 OpenFOAM internalField 文件，前 high_pressure_cells 个为高压，其余为低压。

            参数:
                filename: 输出文件名
                total_cells: 总单元格数量
                high_pressure_cells: 前多少个单元为高压
                high_pressure: 高压值 (Pa)
                low_pressure: 低压值 (Pa)
            """
            if high_pressure_cells > total_cells:
                raise ValueError("高压单元数不能超过总单元数")

            pressures = [high_pressure] * high_pressure_cells + [low_pressure] * (total_cells - high_pressure_cells)
            result = "\n".join(str(p) for p in pressures)
            # 替换括号内的内容，并在右括号后紧跟换行时补上分号
            code_text = re.sub(r'\([^()]*\)', f'({result})', old_code_text)
            return code_text


        input_with_messages = [i.content  for i in with_messages]
        file_list = []

        async_qa_tutorial = AsyncQA_tutorial()
        async_qa = AsyncQA_Ori()
        document_text = self.read_openfoam_tutorials(f"{config_path.Database_PATH}/openfoam_tutorials.txt")    #准备各个文件的写入地方
        allrun_file_path = f'{config_path.Case_PATH}/Allrun'
        postprocessing_python_path = f'{config_path.Case_PATH}/postprocessing_python.py'
        postprocessing_allrun_path = f'{config_path.Case_PATH}/Allrun_postprocessing'
        cfd_task = input_with_messages[0]
        error_log_path = f"{config_path.Case_PATH}/error_log.json"
        if global_statistics.Run_loop == 0:
            # delete error_log
            if os.path.exists(error_log_path):
                os.remove(error_log_path)

        postprocessing_var_path = Path(config_path.Case_PATH) / "postprocessing_var.json"
        allrun_outfile_path = Path(config_path.Case_PATH) / "Allrun.out"
        allrun_postprocessingout_path = Path(config_path.Case_PATH) / "Allrun_postprocessing.out"
        if os.path.exists(postprocessing_var_path) and os.path.isfile(postprocessing_var_path):
            if global_statistics.Executability == 0:
                global_statistics.Executability = 6
        # elif os.path.exists(allrun_postprocessingout_path) and os.path.isfile(allrun_postprocessingout_path):
        #     if global_statistics.Executability == 0:
        #         global_statistics.Executability = 4
        # elif os.path.exists(allrun_outfile_path) and os.path.isfile(allrun_outfile_path):
        #     if global_statistics.Executability == 0:
        #         global_statistics.Executability = 3

        similarity_matrix = self.calculate_similarity(error_log_path)
        if similarity_matrix is not None:
            print('1-2','1-3','2-3')
            print(similarity_matrix[0,1],similarity_matrix[0,2],similarity_matrix[1,2])
            if similarity_matrix[0,2] > 0.9 or similarity_matrix[1,2] > 0.9:
                if config_path.temperature < 0.5:
                    config_path.temperature = 0.5

                elif config_path.temperature == 0.5:
                    if config_path.If_all_files:
                        config_path.If_all_files = False
                    elif config_path.If_RAG:
                        config_path.If_RAG = False
            else:
                config_path.temperature = 0.01
        print('temperature:',config_path.temperature)
        print('If_RAG:',config_path.If_RAG)
        print('If_all_files:',config_path.If_all_files)
        # 读取依赖规则
        with open("/mnt/d/Ubantu_run/MetaOpenFOAM/src/foam_dependencies.yaml", "r") as f:
            rules = yaml.safe_load(f)
        for j,i in enumerate(with_messages[1:]):
            
            if global_statistics.Executability < 3: #这行代码的逻辑到底是什么

                # need to judge whether to write/rewrite allrun_file
                # first wirte allrun: need to do after file generatation
                file_name = self.parse_flie_name(i.content)
                file_list.append(file_name)
                
                IF_rewrite = self.parse_rewirte(i.content)
                # need to judge whether to write/rewrite allrun_file
                
                if 'Allrun' in file_name:
                    
                    allrun_write = "None"

                    if os.path.exists(allrun_file_path) and 'rewrite' not in IF_rewrite:

                        print(f"Allrun file already exists. Skipping...")

                        with open(allrun_file_path, 'r', encoding='utf-8') as allrun_file:

                            allrun_write = allrun_file.read()
                            print('allrun_write2:',allrun_write)

                    elif os.path.exists(allrun_file_path) and 'rewrite' in IF_rewrite:

                        print(f"Allrun file is going to be rewritten...")
                        file_list = self.read_files(config_path.Case_PATH)

                        find_tutorial = self.read_tutorial()
                        #print("find_tutorial:",find_tutorial)
                        case_name = self.get_case_name(find_tutorial)
                        #print("case_name:",case_name)
                        allrun_tutorial = self.get_allrun_tutorial(case_name)

                        promt_allrun_rewrite = self.PROMPT_TEMPLATE_allrun_rewrite.format(requirement=i.content, tutorial = allrun_tutorial)
                        rsp = await async_qa.ask(promt_allrun_rewrite)
                        code_text = self.parse_allrun_new(rsp)
                        print('rewritten allrun file:',code_text)
                        self.save_file(allrun_file_path, code_context=str(code_text))
                        

                else:  # not allrun file                              #似乎命中的不是很准确
                    folder_name = self.parse_folder_name(i.content)
                    IF_rewrite = self.parse_rewirte(i.content)

                    file_path = f"{config_path.Case_PATH}/{folder_name}/{file_name}"
                    case_name_true = os.path.basename(config_path.Case_PATH)
                    
                    if os.path.exists(file_path) and 'rewrite' not in IF_rewrite:
                        print(f"File {file_name} already exists in {folder_name}. Skipping...")
                        continue
                    
                    if config_path.If_RAG:
                        if config_path.tasks >= 3:
                            case_info = self.read_similar_case(f"{config_path.Para_PATH}/find_tutorial.txt")
                        else:
                            case_info = self.read_similar_case(f"{config_path.Case_PATH}/find_tutorial.txt")
                        print('case_info:',case_info)
                        case_name = case_info['case_name']
                        case_domain = case_info['case_domain']
                        case_category = case_info['case_category']
                        case_solver = case_info['case_solver']
                        similar_file = f"```input_file_begin: input {file_name} file of case {case_name} (domain: {case_domain}, category: {case_category}, solver:{case_solver})"
                        
                        tutorial_file = self.find_similar_file(similar_file,document_text)   #找最相似的文件
                        print("tutorial_file:",tutorial_file)
                        if tutorial_file == "None":
                            prompt_find = self.PROMPT_Find.format(file_name=file_name, file_folder=folder_name, case_name = case_name_true)
                            rsp = await async_qa_tutorial.ask(prompt_find)
                            result = rsp["result"]
                            print("find_similar_foamfile:", result)
                            doc = rsp["source_documents"]
                            tutorial_file = doc[0].page_content
                            print("find_tutorial_file:",tutorial_file)

                        print(f"File {file_name} is going to be written")
                        """
                        待开发"""
                        if folder_name == "0":
                           #实现0文件和blockMeshDict的依赖联动
                            with open(f"{config_path.Case_PATH}/system/blockMeshDict", 'r', encoding='utf-8') as blockMeshDict_file:
                                    blockMeshDict_content = blockMeshDict_file.read()
                            if "nonuniform List<scalar>" in tutorial_file:
                                cleaned_tutorial_file = re.sub(r"\([\s\S]*?\)", "()", tutorial_file)
                                prompt = self.PROMPT_Initial_pt_physical_quantity_TEMPLATE.format(requirement=i.content, blockMeshDict_file = blockMeshDict_content, tutorial_file = cleaned_tutorial_file)
                            else:
                                prompt = self.PROMPT_Initial_physical_quantity_TEMPLATE.format(requirement=i.content, blockMeshDict_file = blockMeshDict_content, tutorial_file = tutorial_file)
                                #prompt = self.METAOPENFOAM_PROMPT_TEMPLATE.format(requirement=i.content, tutorial_file = tutorial_file)
                        elif file_name == "topoSetDict":
                            with open(f"{config_path.Case_PATH}/system/controlDict", 'r', encoding='utf-8') as controlDict_file:
                                    controlDict_content = controlDict_file.read()
                            prompt = self.PROMPT_topsetdict_TEMPLATE.format(requirement=i.content, controlDict_content = controlDict_content, tutorial_file = tutorial_file)

                        elif file_name == "funkySetFieldsDict" or  file_name == "setFieldsDict":
                            with open(f"{config_path.Case_PATH}/system/blockMeshDict", 'r', encoding='utf-8') as blockMeshDict_file:
                                    blockMeshDict_content = blockMeshDict_file.read()
                            #读取0文件夹下的所有文件内容
                            zero_dir = os.path.join(config_path.Case_PATH, "0")
                            all_content = []
                            for tmp_filename in sorted(os.listdir(zero_dir)):
                                tmp_file_path = os.path.join(zero_dir, tmp_filename)

                                if not os.path.isfile(tmp_file_path):
                                    continue

                                with open(tmp_file_path, "r", encoding="utf-8") as f:
                                    content = f.read()

                                separator = (
                                    "\n"
                                    + "=" * 60
                                    + f"\n>>> FILE: 0/{tmp_filename}\n"
                                    + "=" * 60
                                    + "\n"
                                )
                                all_content.append(separator + content)
                            merged_content = "\n".join(all_content)
                            prompt = self.SETFIELDS_PROMPT_TEMPLATE.format(requirement=i.content, tutorial_file = tutorial_file, blockdict=blockMeshDict_content, tutorial_zero_files = merged_content)
                        elif file_name == "thermophysicalProperties":
                             results = []
                             for msg in input_with_messages[1:]:
                                text = msg

                                # ① 排除 chemistryProperties 和 thermophysicalProperties
                                if "chemistryproperties" in text or "thermophysicalproperties" in text:
                                    continue

                                # ② 只处理含 chem 或 thermo 的元素
                                if ("chem" in text or "therm" in text) and "constant" in text:
                                    # ③ 提取带 chem/thermo 的单词
                                    words = re.findall(r"\b[a-zA-Z0-9_]*(?:chem|therm)[a-zA-Z0-9_]*\b", text)
                                    for w in words:
                                        results.append(w)

                             # 输出字符串（用空格拼接，也可以用逗号）
                             output = " ".join(results)
                             print(output)
                             prompt = self.SETFIELDS_PROMPT_thermophysicalProperties.format(requirement=i.content, tutorial_file = tutorial_file,chemical_mechanism_file = output)
                        else:
                             prompt = self.PROMPT_TEMPLATE.format(requirement=i.content, tutorial_file = tutorial_file)
                    else:
                        prompt = self.PROMPT_TEMPLATE_no_tutorial.format(requirement=i.content)
                    """
                    为非均匀网格生成定制专门处理流程待开发"""
                    # if file_name == "blockMeshDict":
                    #     prompt = self.PROMPT_blockMeshDict_TEMPLATE.format(requirement=i.content, tutorial_file = tutorial_file)
                    #     parts = tutorial_file.split('\n', 1) 
                    #     code_text = parts[-1]

                    # if file_name == "p" or file_name == "T" or file_name == "U":                                          #为PT文件设置专门的处理流程
                    #     prompt = self.PROMPT_Initial_pt_physical_quantity_TEMPLATE.format(requirement=i.content, blockMeshDict_file = blockMeshDict_content,tutorial_file = tutorial_file)
                    #     rsp = await async_qa.ask(prompt)
                    #     code_text = self.parse_context(rsp)
                    #     if "nonuniform List<scalar>" in code_text :                       
                    #         initial_parameters = self.PROMPT_Initial_physical_quantity_TEMPLATE2.format(requirement=i.content, file=code_text, tutorial_file=tutorial_file)
                    #         rsp = await async_qa.ask(initial_parameters)
                    #         # 用正则提取最外层花括号及其中内容
                    #         match = re.search(r'\{[\s\S]*\}', rsp)

                    #         if match:
                    #             rsp = match.group(0)
                    #             initial_parameters_dic = json.loads(rsp)  # 转为字典
                    #             print(initial_parameters_dic)
                    #         else:
                    #             print("没有找到合法 JSON")
                    #         code_text = generate_internal_field(code_text,total_cells = initial_parameters_dic["total_cells"], 
                    #                                             high_pressure_cells = initial_parameters_dic["high_pressure_cells"],
                    #                                             high_pressure = initial_parameters_dic["high_pressure"], 
                    #                                             low_pressure = initial_parameters_dic["low_pressure"] )
                    #为机理文件设置专门的处理流程机理文件过于复杂，直接用已有的是最好的
                    if folder_name == "constant" and file_name not in ["chemistryProperties", "thermophysicalProperties"] and ("chem" in file_name.lower() or "therm" in file_name.lower()):
                        parts = tutorial_file.split('\n', 1)
                        code_text = parts[-1]
                    else:
                        # 多模态输入部分
                        blockmeshdict_path = os.environ.get("BLOCKMESHDICT_PATH", "")
                        setfields_path = os.environ.get("SETFIELDS_PATH", "")
                        if file_name == "blockMeshDict" and os.path.exists(blockmeshdict_path):
                            with open(blockmeshdict_path, 'r', encoding='utf-8') as file:
                                code_text = file.read()
                        elif file_name == "funkySetFieldsDict" and os.path.exists(setfields_path):
                            with open(setfields_path, 'r', encoding='utf-8') as file:
                                code_text = file.read()
                           # 一般的文件写入流程
                        else:
                            rsp = await async_qa.ask(prompt)
                            code_text = self.parse_context(rsp)
                    print('folder_name',folder_name)
                    print('file_name',file_name)
                    # if file_name == "chemistryProperties" and "refmapping" not in code_text:
                    #     # 要插入的新字段内容
                    #     new_block = """
                    #     refmapping\n{\n}
                    #     """
                    #     # 使用正则匹配 odeCoeffs 的整个块（贪婪匹配）
                    #     pattern = r"(odeCoeffs\s*\{[^}]*\})"

                    #     # 在 odeCoeffs 后插入新的字段块
                    #     code_text = re.sub(pattern, r"\1\n" + new_block, code_text, flags=re.DOTALL)
                    if folder_name.strip() and file_name.strip():
                        patterns = [
                            "input_file_end.",
                            "```",
                            "input_file_end",
                            "input_file_end\n",
                            "begin:",
                            "input_file_end.\n",
                        ]

                        code_text = str(code_text)

                        for p in patterns:
                            code_text = code_text.replace(p, "")

                        code_text = code_text.strip()  # 最后只去掉空白字符（安全）
                        self.save_file(file_path, code_context= code_text)
                    else:
                        print("Folder name or file name is empty, skipping save operation.")
                    # if file_name == "blockMeshDict":
                    #     import subprocess
                    #     script_path = f'bash -c "source /home/xupan/OpenFOAM/OpenFOAM-9/etc/bashrc && source $WM_PROJECT_DIR/bin/tools/RunFunctions && blockMesh"'
                    #     result = subprocess.run(script_path, shell=True, cwd=config_path.Case_PATH, capture_output=True, text=True)
                    #     #output = subprocess.check_output("blockMesh", shell=True, text=True)
                    #     runtimes = 1
                    #     while((result.returncode !=0 or " FOAM Warning" in result.stdout) and runtimes < 5 ):
                    #         runtimes += 1
                    #         print("blockMesh error, retrying...")
                    #         prompt = self.PROMPT_blockMeshDict_Modify_TEMPLATE.format(requirement=i.content, blockDict = str(code_text) , Questions = result.stdout)
                    #         rsp = await async_qa.ask(prompt)
                    #         code_text = self.parse_context(rsp)
                    #         print('folder_name',folder_name)
                    #         print('file_name',file_name)
                    #         if folder_name.strip() and file_name.strip():
                    #             self.save_file(file_path, code_context=str(code_text))
                    #         else:
                    #             print("Folder name or file name is empty, skipping save operation.")
                    #         result = subprocess.run(script_path, shell=True, cwd=config_path.Case_PATH, capture_output=True, text=True)

            elif global_statistics.Executability == 3 and os.path.exists(postprocessing_allrun_path) and config_path.tasks>=2:
                print('rewrite for Allrun_postprocessing')
                error_log_path = f"{config_path.Case_PATH}/error_postprocessing_log.json"
                if global_statistics.Postprocess_loop == 0:
                    # delete error_log
                    if os.path.exists(error_log_path):
                        os.remove(error_log_path)

                similarity_matrix = self.calculate_similarity(error_log_path)
                if similarity_matrix is not None:
                    print('1-2','1-3','2-3')
                    print(similarity_matrix[0,1],similarity_matrix[0,2],similarity_matrix[1,2])
                    if similarity_matrix[0,2] > 0.9 or similarity_matrix[1,2] > 0.9:
                        config_path.temperature = 0.5
                    else:
                        config_path.temperature = 0.01
                
                promt_allrun_postprocessing_rewrite = i
                if promt_allrun_postprocessing_rewrite != "None":
                    rsp = await async_qa.ask(promt_allrun_postprocessing_rewrite)
                    print('Executability = 3, to rewrite postprocessing:',rsp)
                    
                    Allrun_postprocessing = self.parse_post_processing_new(rsp)
                    print('Allrun_postprocessing:',Allrun_postprocessing)
                    Allrun_postprocessing_path = f'{config_path.Case_PATH}/Allrun_postprocessing'

                    self.save_file(Allrun_postprocessing_path, code_context=str(Allrun_postprocessing))
                
            elif global_statistics.Executability == 3 and not os.path.exists(postprocessing_allrun_path) and config_path.tasks>=2:    #写后处理脚本的转成VTK

                config_path.temperature = 0.0
                print('temperature:',config_path.temperature)
                CFD_task, CFD_post_tasks = self.load_CFD_tasks(f"{config_path.Case_PATH}/CFD_tasks.json")
                #controlDict_text = self.read_controlDict(config_path.Case_PATH)
                postprocessing_list = self.read_postprocess_commands(config_path.Database_PATH)
                #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], controlDict_text=controlDict_text, postprocessing_list = postprocessing_list)
                #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_allrun_JSON.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], postprocessing_list = postprocessing_list, JSON_allrun_postprocessing = JSON_allrun_postprocessing)
                end_time = self.get_end_time(config_path.Case_PATH)
                print('endTime',end_time)
                
                end_Time_file_list = self.get_files_in_endTime(end_time)

                json_structure = """
                {
                    "file_names": ["specific_file_name1", "specific_file_name2", ...]
                }
                """
                prompt_postprocessing_if_exist = self.PROMPT_TEMPLATE_postprocessing_if_exist.format(CFD_task=CFD_task, CFD_postprocessing_task = CFD_post_tasks, file_list = end_Time_file_list,json_structure = json_structure)
                rsp = await async_qa.ask(prompt_postprocessing_if_exist)
                if 'No' in rsp:
                    prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_allrun3.format(CFD_task=CFD_task, CFD_postprocessing_task = CFD_post_tasks, postprocessing_list = postprocessing_list)
                else:
                    json_match = re.search(r'```json(.*)```', rsp, re.DOTALL)
                    json_str = json_match.group(1) if json_match else None
                    if json_str:
                        related_file_list = self.parse_json_output(json_str)
                        print("related_file_list:", related_file_list)
                    else:
                        print("No JSON found in the provided text.")
                        sys.exit()

                    prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_allrun_vtk.format(CFD_task=CFD_task, CFD_postprocessing_task = CFD_post_tasks, related_file_list = related_file_list)
                
                #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_allrun2.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], postprocessing_list = postprocessing_list)
                print('prompt_postprocessing:',prompt_postprocessing)

                rsp = await async_qa.ask(prompt_postprocessing)
                print('postprocessing_rsp:',rsp)

                Allrun_postprocessing = self.parse_post_processing_new(rsp)
                #print('Allrun_postprocessing:',Allrun_postprocessing)

                Allrun_postprocessing_path = f'{config_path.Case_PATH}/Allrun_postprocessing'

                self.save_file(Allrun_postprocessing_path, code_context=str(Allrun_postprocessing))

            elif (global_statistics.Executability == 4 or global_statistics.Executability == 5) and os.path.exists(postprocessing_python_path) and config_path.tasks>=2:
                
                print('rewrite for Allrun_postprocessing and postprocessing_python')

                error_log_path = f"{config_path.Case_PATH}/postprocessing_python_error.json"
                if global_statistics.Postprocess_loop == 0:
                    # delete error_log
                    if os.path.exists(error_log_path):
                        os.remove(error_log_path)

                similarity_matrix = self.calculate_similarity(error_log_path)
                if similarity_matrix is not None:
                    print('1-2','1-3','2-3')
                    print(similarity_matrix[0,1],similarity_matrix[0,2],similarity_matrix[1,2])
                    if similarity_matrix[0,2] > 0.9 or similarity_matrix[1,2] > 0.9:
                        config_path.temperature = 0.5
                    else:
                        config_path.temperature = 0.01
                promt_postprocessing_rewrite = i
                rsp = await async_qa.ask(promt_postprocessing_rewrite)
                
                print('Executability = 4/5, to rewrite postprocessing:',rsp)

                # save new python file

                python_script = self.parse_python_new(rsp)

                self.save_file(postprocessing_python_path, code_context=str(python_script))
                # Allrun_postprocessing = self.parse_Modified_post_processing(rsp)
                # print('Allrun_postprocessing:',Allrun_postprocessing)
                # Allrun_postprocessing_path = f'{config_path.Case_PATH}/Allrun_postprocessing'

                # self.save_file(Allrun_postprocessing_path, code_context=str(Allrun_postprocessing))

                # postprocessing_python = self.parse_Modified_python(rsp)
                
                # print('postprocessing_python:',postprocessing_python)
                # self.save_file(postprocessing_python_path, code_context=str(postprocessing_python))
                # prompt_python_env = self.PROMPT_TEMPLATE_python_env.format(python_text = postprocessing_python)
                # print('prompt_python_env:',prompt_python_env)
                # rsp = await async_qa.ask(prompt_python_env) 
                # print('rsp_python_env:',rsp)
                # python_env = self.parse_python_env(rsp)
                # print('parse_python_env:',python_env)
                # libraries = re.findall(r'\b\w+\b', python_env.strip())
                # print('python_env:',libraries)
                # # Check for each package and install if necessary
                # for package in libraries:
                #     try:
                #         __import__(package)
                #         print(f"{package} is already installed.")
                #     except ImportError:
                #         print(f"{package} is not installed. Installing...")
                #         self.install_package(package)

                # print("All required packages are installed.")
                
                
                #global_statistics.Executability = 3
                
            elif (global_statistics.Executability == 4 or global_statistics.Executability == 5) and not os.path.exists(postprocessing_python_path) and config_path.tasks>=2:
                config_path.temperature = 0.01
                print('first generate python script')
                CFD_task, CFD_post_tasks = self.load_CFD_tasks(f"{config_path.Case_PATH}/CFD_tasks.json")
                #controlDict_text = self.read_controlDict(config_path.Case_PATH)
                postprocessing_data_path = f"{config_path.Case_PATH}/new_files_data.json"
                all_new_files, new_file_paths = self.load_new_files_data(postprocessing_data_path)
                postprocessing_list = self.read_postprocess_commands(config_path.Database_PATH)
                
                JSON_dependent_var = """ 
                    {
                        "postprocessing_var": "specific_value"
                    }
                """
                print("JSON_dependent_var:",JSON_dependent_var)
                print('new_file_paths:',new_file_paths)
                print('all_new_files:',all_new_files)
                # 目前只考虑单个new_file_paths，即后处理只增加单个文件
                #postprocessing_data = self.read_first_50_lines(new_file_paths[0])

                postprocessing_command = self.read_postprocessing_command()
                prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_python_for_vtk.format(CFD_task=CFD_task, 
                                                                          CFD_postprocessing_task = CFD_post_tasks,
                                                                          postprocessing_command = postprocessing_command,
                                                                          postprocessing_new_data_path = new_file_paths,
                                                                          JSON_dependent_var = JSON_dependent_var)

                #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], controlDict_text=controlDict_text, postprocessing_list = postprocessing_list)
                # prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_python.format(CFD_task=CFD_task, 
                #                                                                           postprocessing_command = postprocessing_command, 
                #                                                                           postprocessing_new_data_path = new_file_paths[0], 
                #                                                                           postprocessing_data = postprocessing_data, 
                #                                                                           JSON_dependent_var = JSON_dependent_var)
                
                print('prompt_postprocessing_python:',prompt_postprocessing)
                rsp = await async_qa.ask(prompt_postprocessing) 
                print('postprocessing_rsp:',rsp)
                posrprocessing_python_script = self.parse_python_new(rsp)
                print('posrprocessing_python_script:',posrprocessing_python_script)
                self.save_file(postprocessing_python_path, code_context=str(posrprocessing_python_script))
            elif global_statistics.Executability == 6 and config_path.tasks>=2:
                print('specific_case already run,EXE = ',global_statistics.Executability)
                
            else:
                print('wrong Executability!!')
                sys.exit()
                

        #first write allrun        
        if not os.path.exists(allrun_file_path):
            #write allrun
            requirement = cfd_task # need to be fixed
            script_path = f'bash -c "source /home/xupan/OpenFoam6/OpenFOAM-6/etc/bashrc && source $WM_PROJECT_DIR/bin/tools/RunFunctions && blockMesh && funkySetFields -time 0"'
            result = subprocess.run(script_path, shell=True, cwd=config_path.Case_PATH, capture_output=True, text=True)
                # 检查执行结果
            if result.returncode == 0:
                print("funkysetFields 命令执行成功")
                print("输出:", result.stdout)
            else:
                print("funkysetFields 命令执行失败")
                print("错误:", result.stderr)
                print("返回码:", result.returncode)
            
            script_path = f'bash -c "source /home/xupan/OpenFoam6/OpenFOAM-6/etc/bashrc && source $WM_PROJECT_DIR/bin/tools/RunFunctions && blockMesh && topoSet && setFields"'
            result = subprocess.run(script_path, shell=True, cwd=config_path.Case_PATH, capture_output=True, text=True)
                # 检查执行结果
            if result.returncode == 0:
                print("setFields 命令执行成功")
                print("输出:", result.stdout)
            else:
                print("setFields 命令执行失败")
                print("错误:", result.stderr)
                print("返回码:", result.returncode)
            async_qa_allrun = AsyncQA_allrun()
            runlists = ['isTest', 'getNumberOfProcessors','getApplication','runApplication','runParallel','compileApplication','cloneCase','cloneMesh']
            commands = self.read_commands(config_path.Database_PATH)
            file_list = self.read_files(config_path.Case_PATH)

            find_tutorial = self.read_tutorial()
            #print("find_tutorial:",find_tutorial)
            case_name = self.get_case_name(find_tutorial)
            #print("case_name:",case_name)
            allrun_tutorial = self.get_allrun_tutorial(case_name)
            #print("allrun_tutorial:",allrun_tutorial)
            commands.extend(["detonationEulerFoam","detonationNSFoam_mixtureAverage","detonationNSFoam_Sutherland","DLB","dynamicMesh2D","fluxSchemes","decomposePar","reactingrhoCentralFoam_rde_TianheVersion"])
            with open(f"{config_path.Case_PATH}/system/controlDict", 'r', encoding='utf-8') as controlDict_file:
                    controlDict_content = controlDict_file.read()
            with open(config_path.Case_PATH + '/system/decomposeParDict', 'r', encoding='utf-8') as decomposeParDict_file:
                    decomposeParDict_content = decomposeParDict_file.read()        
            prompt_allrun = self.PROMPT_TEMPLATE_allrun.format(
                requirement=requirement, 
                tutorial = allrun_tutorial,
                file_list = file_list, 
                commands = commands, 
                runlists = runlists,
                decomposeParDict = decomposeParDict_content,
                controlDict = controlDict_content
                )
            # prompt_allrun = self.metaopenfoam_PROMPT_TEMPLATE_allrun.format(
            #     requirement=requirement, 
            #     tutorial = allrun_tutorial,
            #     file_list = file_list, 
            #     commands = commands, 
            #     runlists = runlists)
            
            
            rsp = await async_qa_allrun.ask(prompt_allrun) 
            result = rsp["result"]
            #doc = rsp["source_documents"]
            #print("allrun_source_documents:",doc[0])
            #print("allrun:",result)
            allrun_write = self.parse_allrun_new(result)
            with open(allrun_file_path, 'w') as outfile:  
                outfile.write(allrun_write.strip("\n"))

            print('allrun_write:',allrun_write)
        #first write postprocessing in controlDict
        # 在OpenFOAM的{case_name}模拟中，想要后处理提取{dependent_var}，请分析如何在controlDict中写后处理程序并在运行后得到的postprocessing中利用python程序自动分析得到{dependent_var}
        # v2: 在controlDict中会遇到在run的阶段将后处理改没的情况，所以还是写在allrun中吧

        # if not os.path.exists(postprocessing_allrun_path) and global_statistics.Executability == 3:
        #     CFD_task, independent_vars, dependent_vars, samples, Specific_CFD_tasks, Multi_CFD_tasks = self.load_parameters(config_path.Para_PATH)
        #     #controlDict_text = self.read_controlDict(config_path.Case_PATH)
        #     postprocessing_list = self.read_postprocess_commands(config_path.Database_PATH)
        #     #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], controlDict_text=controlDict_text, postprocessing_list = postprocessing_list)
        #     prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_allrun.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], postprocessing_list = postprocessing_list)
        #     print('prompt_postprocessing:',prompt_postprocessing)
        #     rsp = await async_qa.ask(prompt_postprocessing)
        #     print('postprocessing_rsp:',rsp)

        #     Allrun_postprocessing = self.parse_post_processing(rsp)
        #     print('Allrun_postprocessing:',Allrun_postprocessing)

        #     Allrun_postprocessing_path = f'{config_path.Case_PATH}/Allrun_postprocessing'

        #     self.save_file(Allrun_postprocessing_path, code_context=str(Allrun_postprocessing))

            
        # if not os.path.exists(postprocessing_python_path) and global_statistics.Executability == 4:
            
        #     CFD_task, independent_vars, dependent_vars, samples, Specific_CFD_tasks, Multi_CFD_tasks = self.load_parameters(config_path.Para_PATH)
        #     #controlDict_text = self.read_controlDict(config_path.Case_PATH)
        #     postprocessing_data_path = f"{config_path.Case_PATH}/new_files_data.json"
        #     all_new_files, new_file_paths = self.load_new_files_data(postprocessing_data_path)
        #     postprocessing_list = self.read_postprocess_commands(config_path.Database_PATH)
        #     dependent_var=dependent_vars[0]
        #     JSON_dependent_var = f"""
        #         {
        #             {dependent_var}: specific_value,
        #         }
        #     """
            
        #     postprocessing_data = self.read_first_30_lines(postprocessing_data_path)
            
        #     #prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], controlDict_text=controlDict_text, postprocessing_list = postprocessing_list)
        #     prompt_postprocessing = self.PROMPT_TEMPLATE_postprocessing_python.format(CFD_task=CFD_task, dependent_var=dependent_vars[0], postprocessing_data_path = postprocessing_data_path, postprocessing_data = postprocessing_data, JSON_dependent_var = JSON_dependent_var)
            
        #     print('prompt_postprocessing:',prompt_postprocessing)
        #     rsp = await async_qa.ask(prompt_postprocessing) 
        #     print('postprocessing_rsp:',rsp)
        #     sys.exit()
            
            
        #     Allrun_postprocessing = self.parse_post_processing(rsp)
        #     print('Allrun_postprocessing:',Allrun_postprocessing)
        #     Allrun_postprocessing_path = f'{config_path.Case_PATH}/Allrun_postprocessing'

        #     self.save_file(Allrun_postprocessing_path, code_context=str(Allrun_postprocessing))

        #     postprocessing_python = self.parse_python(rsp)
            
        #     print('postprocessing_python:',postprocessing_python)
        #     self.save_file(postprocessing_python_path, code_context=str(postprocessing_python))
        #     prompt_python_env = self.PROMPT_TEMPLATE_python_env.format(python_text = postprocessing_python)
        #     print('prompt_python_env:',prompt_python_env)
        #     rsp = await async_qa.ask(prompt_python_env) 
        #     print('rsp_python_env:',rsp)
        #     python_env = self.parse_python_env(rsp)
        #     print('parse_python_env:',python_env)
        #     libraries = re.findall(r'\b\w+\b', python_env.strip())
        #     print('python_env:',libraries)
        #     # Check for each package and install if necessary
        #     for package in libraries:
        #         try:
        #             __import__(package)
        #             print(f"{package} is already installed.")
        #         except ImportError:
        #             print(f"{package} is not installed. Installing...")
        #             self.install_package(package)

        #     print("All required packages are installed.")

        return "dummpy message"

    
    @staticmethod
    def parse_flie_name(rsp):
        pattern = r"一个 [oO][pP][eE][nN][fF][oO][aA][mM] (.*) 文件"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_flie = match.group(1) if match else ''
        return your_task_flie

    @staticmethod
    def parse_folder_name(rsp):
        pattern = r"在 (.*) 文件夹"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else ''
        return your_task_folder
      
    @staticmethod
    def parse_context(rsp):
        pattern = r"(FoamFile.*?)(?:```|$)"
        match = re.search(pattern, rsp, re.DOTALL)
        if match:
            your_task_flie = match.group(1) 
        else:
            match2 = re.search(r'```(?:.*?\n)(.*?)\n```', rsp, re.DOTALL)
            if match2:
                your_task_flie =  match2.group(1) 
            else:
                your_task_flie = rsp
        return your_task_flie
    
    @staticmethod
    def parse_rewirte(rsp):
        pattern = r"to (.*) a OpenFoam"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_flie = match.group(1) if match else ''
        return your_task_flie
    def find_similar_file(self, start_string, document_text):
        ref_pattern = r"input_file_begin:\s*input\s+([^\s]+)\s+file\s+of\s+case\s+([^\s(]+)"
        match = re.search(ref_pattern, start_string)
        file_name, case_name = match.group(1), match.group(2)
        if "chemistryProperties" not in start_string and "thermophysicalProperties" not in start_string and "fvSchemes" not in start_string and ("chem" in start_string.lower() or "therm" in start_string.lower()): 
           target_pattern = rf"input_file_begin:\s*input\s+([^\s]+)\s+file\s+of\s+case\s+{re.escape(case_name)}"##匹配这和算例下所有文件
           if "chem" in start_string.lower():
               for m in re.finditer(target_pattern, document_text):
                   fname = m.group(1)
                   if "chem" in fname.lower() and fname not in ("chemistryProperties", "thermophysicalProperties","fvSchemes"):
                        start_pos = m.start()
                        break
           elif "therm" in start_string.lower():
                for m in re.finditer(target_pattern, document_text):
                   fname = m.group(1)
                   if "therm" in fname.lower() and fname not in ("chemistryProperties", "thermophysicalProperties","fvSchemes"):
                        start_pos = m.start()
                        break
        else:
            start_pos = document_text.find(start_string)
        if start_pos == -1:
            return "None"
        
        end_pos = document_text.find("input_file_end.", start_pos)
        if end_pos == -1:
            return "None"
        
        return document_text[start_pos:end_pos + len("input_file_end.")]
    def read_similar_case(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # 初始化要读取的字段
                case_info = {
                    'case_name': None,
                    'case_domain': None,
                    'case_category': None,
                    'case_solver': None
                }
                
                for line in file:
                    line = line.strip()
                    if line.startswith('case name:'):
                        case_info['case_name'] = line.split('case name:')[1].strip()
                    elif line.startswith('case domain:'):
                        case_info['case_domain'] = line.split('case domain:')[1].strip()
                    elif line.startswith('case category:'):
                        case_info['case_category'] = line.split('case category:')[1].strip()
                    elif line.startswith('case solver:'):
                        case_info['case_solver'] = line.split('case solver:')[1].strip()

                return case_info
            
        except FileNotFoundError:
            return f"file {file_path} not found"
    
    # def read_similar_case(self, file_path):
    #     try:
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             # 读取文件的第一行
    #             first_line = file.readline().strip()
                
    #             case_name_pos = first_line.find('case name:')
    #             if case_name_pos == -1:
    #                 return "None"
                
    #             case_name = first_line[case_name_pos + len('case name:'):].strip()
    #             return case_name
        
    #     except FileNotFoundError:
    #         return f"file {file_path} not found"
    def read_openfoam_tutorials(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:

                content = file.read()
                return content
        except FileNotFoundError:
            return f"file {file_path} not found"
        except Exception as e:
            return f"reading file meet error: {e}"
    def save_file(self, file_path: str, code_context: str) -> None:

        directory = os.path.dirname(file_path)
        # Create the folder path if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        with open(file_path, 'w') as file:
            file.write(code_context)  # 将代码写入文件

        print(f"File saved successfully at {file_path}")
        

    
        
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
    def read_tutorial(self):
        if config_path.tasks>=3:
            save_path = config_path.Para_PATH
        else:
            save_path = config_path.Case_PATH
        file_path = f"{save_path}/find_tutorial.txt"
        with open(file_path, 'r') as file_handle:
            content = file_handle.read() 
        return content
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
    def install_package(self,package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
    def load_parameters(self, file_path):
        with open(Path(file_path) / "Parameter.txt", "r") as file:
            parameters = json.load(file)
        
        CFD_task = parameters["CFD_task"]
        independent_vars = parameters["independent_vars"]
        dependent_vars = parameters["dependent_vars"]
        samples = parameters["samples"]
        Specific_CFD_tasks = parameters["Specific_CFD_tasks"]
        Multi_CFD_tasks = parameters["Multi_CFD_tasks"]
        
        return CFD_task, independent_vars, dependent_vars, samples, Specific_CFD_tasks, Multi_CFD_tasks
    
    def load_CFD_tasks(self, file_path):
        """
        Loads JSON data from a specified file path.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            dict: The loaded JSON data as a dictionary, or None if an error occurs.
        """
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            print(f"JSON data successfully loaded from {file_path}")
            CFD_task = data["CFD_simulation_task"]
            CFD_post_task = data["CFD_postprocessing_task"]
            return CFD_task, CFD_post_task
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def read_controlDict(self, file_path):
        
        controlDict_path = Path(file_path) / "system" / "controlDict"
        if controlDict_path.exists():
            with open(controlDict_path, "r") as file:
                controlDict_text = file.read()
            return controlDict_text
        else:
            print("controlDict file not found.")
            sys.exit()
            return None

    def read_postprocess_commands(self, file_path):
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
            
    def calculate_similarity(self, json_file_path):
        # 检查文件是否存在且包含足够的数据
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                error_log = json.load(file)
                # 读取最新3次的迭代内容
                latest_errors = list(error_log.values())[-3:]
                if len(latest_errors) < 3:
                    print("Not enough data for similarity calculation.")
                    return None
                # 使用TF-IDF向量化文本
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(latest_errors)
                # 计算相似性
                cos_sim = cosine_similarity(tfidf_matrix)
                return cos_sim
        else:
            print("Error log file does not exist.")
            return None
        
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
    
    def read_first_50_lines(self, file_path):
        # Read the file and keep only the first 30 lines as a single string
        with open(file_path, "r") as file:
            lines = file.readlines()
        
        # Join the first 30 lines into a single string
        first_50_lines = "".join(lines[:50])
        return first_50_lines
    
    def read_postprocessing_command(self):
        """读取 Allrun_postprocessing 文件内容"""
        file_path = f"{config_path.Case_PATH}/Allrun_postprocessing"
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return f"Error: The file at {file_path} was not found."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"
        
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
    
    def get_files_in_endTime(self, end_time):
        # 构造文件夹路径
        folder_path = f"{config_path.Case_PATH}/{end_time}"
        
        # 检查文件夹是否存在
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # 获取文件夹中所有文件（不包含子文件夹中的文件）
            file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            return file_list
        else:
            sys.exit()
            return [] 
    
    def parse_json_output(self, json_output):
        """
        Parse the given JSON string to extract the file names.
        
        Args:
            json_output (str): JSON string in the format:
                {
                    "file_names": ["specific_file_name1", "specific_file_name2", ...]
                }
        
        Returns:
            list: A list of file names if the JSON is valid and contains 'file_names'.
                Returns an empty list if the key does not exist or the JSON is invalid.
        """
        try:
            # Parse the JSON string
            data = json.loads(json_output)
            
            # Extract 'file_names' if present
            if "file_names" in data and isinstance(data["file_names"], list):
                return data["file_names"]
            else:
                return []  # Return empty list if 'file_names' key is missing or invalid
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return []  # Return empty list if JSON is invalid

    @staticmethod
    def parse_post_processing(rsp):
        pattern = r"`Allrun_postprocessing` file begin ```(.*)``` `Allrun_postprocessing` file end"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    @staticmethod
    def parse_post_processing_new(rsp):
        pattern = r"```Allrun_postprocessing(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    @staticmethod
    def parse_python(rsp):
        pattern = r"Python script begin ```(.*)``` Python script end"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    @staticmethod
    def parse_python_new(rsp):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    @staticmethod
    def parse_python_env(rsp):
        pattern = r"Python env list begin ```(.*)``` Python env list end"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    
    @staticmethod
    def parse_Modified_post_processing(rsp):
        pattern = r"Modified `Allrun_postprocessing` file begin ```(.*)``` Modified `Allrun_postprocessing` file end"
        match = re.search(pattern, rsp, re.DOTALL)
        if match:
            your_task_folder = match.group(1) 
        else:
            pattern = r"```(.*)```"
            match = re.search(pattern, rsp, re.DOTALL)
            your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    @staticmethod
    def parse_Modified_python(rsp):
        pattern = r"Modified Python script begin ```(.*)``` Modified Python script end"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
    
    @staticmethod
    def parse_allrun_new(rsp):
        pattern = r"```bash(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        your_task_folder = match.group(1) if match else 'None'
        return your_task_folder
        