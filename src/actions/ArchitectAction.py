
import re
from typing import List
import os
from metagpt.actions import Action

from metagpt.schema import Message
from metagpt.logs import logger

from qa_module import AsyncQA_tutorial_name
import config_path
from Statistics import global_statistics
import sys
class ArchitectAction(Action):

    PROMPT_TEMPLATE_divide_task: str = """
    User requirement:
    <{requirement}>
    Your task is to generate the openfoam input foamfiles list following file structure of OpenFOAM cases to meet the user requirements.
    Here is a openfoam case similar to the user requirements
    The following is a case of openfoam:
    <{tutorial}>
    you can take this case as a reference. 
    First, check which files are required in the system folder.
    Then, identify the necessary files in the constant folder.
    Finally, based on the user's simulation requirements, determine which fields need to be initialized in the 0 folder.
    generate the openfoam input foamfiles list following file structure of OpenFOAM cases to meet the user requirements.
    You should split the list of foamFiles into several subtasks, and each subtask should correspond to only one input foamFile.
    Do not allow any subtask to correspond to multiple files.
    Return ```splits into number_of_subtasks subtasks:  
    subtask1: to Write a OpenFoam specific_file_name foamfile in specific_folder_name folder that could be used to meet user requirement:{requirement}.
    subtask2: to Write a OpenFoam specific_file_name foamfile in specific_folder_name folder that could be used to meet user requirement:{requirement}.
    ...

    ``` with NO other texts,
    your subtasks:
    """
    PROMPT_TEMPLATE_divide_task2: str = """
    用户需求：\n<{requirement}>\n你的任务是按照 OpenFOAM 算例的文件结构，生成满足用户需求的 OpenFOAM 输入文件（foamfiles）列表。这里有一个与用户需求高度类似的，已经跑通的OpenFOAM 算例：\n<{tutorial}>\n
    该算例每个文件都是必要的，你需要借鉴这个算例，按照 OpenFOAM 算例的文件结构，生成满足用户需求的 OpenFOAM 输入文件列表。你应该将 foamFiles 列表拆分为多个子任务，每个子任务仅对应一个输入 foamFile。
    不允许任何子任务的描述中提到要写多份文件，在system文件夹中blockdict和contorcldict文件应该优先完成。funkySetFieldsDict、setFieldsDict文件需要放在所有任务的最后面。
    返回 ``` 拆分为 number_of_subtasks 个子任务：
    子任务 1：在 specific_folder_name 文件夹中编写一个 OpenFoam specific_file_name 文件。
    子任务 2：在 specific_folder_name 文件夹中编写一个 OpenFoam specific_file_name 文件。
    ``` 不要有其他文本，你的子任务：\n"""










    PROMPT_TEMPLATE3: str = """
    User requirement:
    {requirement}
    Your task is to generate the openfoam input foamfiles list following file structure of OpenFOAM cases to meet the user requirements.
    You should splits foamfiles list into several subtasks, and one subtask corresponds to one input foamfile
    Return ```splits into number_of_subtasks subtasks:  
    subtask1: to Write a OpenFoam specific_file_name foamfile in specific_folder_name folder that could be used to meet user requirement:{requirement}
    subtask2: to Write a OpenFoam specific_file_name foamfile in specific_folder_name folder that could be used to meet user requirement:{requirement}
    ...

    ``` with NO other texts,
    your subtasks:
    """
    PROMPT_TEMPLATE_Find_case: str = """
    Your task is to find the case that is most similar to the user's requirement:
    {requirement}
    Your task is to generate the openfoam input foamfiles list following file structure of OpenFOAM cases to meet the user requirements.
    You should splits foamfiles list into several subtasks, and one subtask corresponds to one input foamfile
    Return ```
    case name: specific_case_name
    case domain: specific_case_domain
    case category: specific_case_category
    case solver: specific_case_solver
    file_names: specific_file_names
    file_folders: specific_file_folders
    ...

    ``` with NO other texts
    """
    PROMPT_Translate: str = """
        Translate the following user request into the specified standard format:
        User request:
        {requirement}
        Standard format:
        case name: specific_case_name
        case domain: specific_case_domain
        case category: specific_case_category
        case solver: specific_case_solver
        Note that case domain could only be one of following strings:
        [basic, compressible, discreteMethods, DNS, electromagnetics, financial, heatTransfer, incompressible, lagrangian, mesh, multiphase, stressAnalysis]
    """

    PROMPT_Find: str = """
        Find the OpenFOAM case that most closely matches the following case:
        {user_case}
        where case domain, case category and case solver should be matched with the highest priority
    """

    name: str = "ArchitectAction"

    async def run(self, with_messages:List[Message]=None, **kwargs) -> List[str]:
        
        async_qa_tutotial = AsyncQA_tutorial_name()

        print('CFD task:',with_messages.content)
        # prompt_Translate = self.PROMPT_Translate.format(requirement=with_messages.content)
        # rsp = await async_qa_tutotial.ask(prompt_Translate)
        # user_case = rsp["result"]
        # print('user_case:',user_case)
        # #case_name = self.parse_case_name(user_case)
        # # if config_path.run_times > 1:
        # #     config_path.Case_PATH = f"{config_path.Run_PATH}/{case_name}_{global_statistics.runtimes}"
        # # else:
        # #     config_path.Case_PATH = f"{config_path.Run_PATH}/{case_name}"
        # # os.makedirs(config_path.Case_PATH, exist_ok=True)

        # prompt_Find = self.PROMPT_Find.format(user_case=user_case)
        # rsp = await async_qa_tutotial.ask(prompt_Find)
        # doc = rsp["source_documents"]
        # tutorial = doc[0]
        # print('find_case',tutorial)
        # save_path = config_path.Case_PATHs[i]

        #self.save_find_tutorial(tutorial.page_content, save_path)
        if config_path.tasks>=3:                            #？？？？？？
            tutorial = self.read_tutorial(config_path.Para_PATH)
        else:
            tutorial = self.read_tutorial(config_path.Case_PATH)
        """
        生成文件写入任务
        多少个文件对应多少任务"""      
        prompt_subtask = self.PROMPT_TEMPLATE_divide_task2.format(requirement=with_messages.content, tutorial=tutorial)
        rsp = await async_qa_tutotial.ask(prompt_subtask)
        result = rsp["result"]
        logger.info(str(result))
        

        subtasks: List[str] = self.split_subtask(result)
        subtasks = [i + "以满足用户需求：" +  with_messages.content + "。"for i in subtasks]

        return subtasks
    
    def split_subtask(self, content: str) -> list:

        header_pattern = re.compile(r'拆分为\s*(\d+)\s*个子任务：')

        subtask_pattern = re.compile(r'(子任务\s*\d+\s*[：:]\s*.*?)(?=子任务\s*\d+\s*[：:]|$)',re.DOTALL)

        header_match = header_pattern.search(content)
        if header_match:
            number_of_subtasks = int(header_match.group(1))
        else:
            return []

        subtasks = []

        for match in subtask_pattern.finditer(content):

            subtasks.append(match.group(1))

        if len(subtasks) != number_of_subtasks:

            print("Warning: Declared number of subtasks does not match extracted subtasks.")
        
        return subtasks
    @staticmethod
    def parse_case_name(rsp):
        match = re.search(r'case name:\s*(.+)', rsp)
        your_task_folder = match.group(1).strip() if match else 'None'
        return your_task_folder


    def read_tutorial(self, read_path):
        file_path = f"{read_path}/find_tutorial.txt"
        with open(file_path, 'r', encoding='utf-8') as file:
            tutorial = file.read()

        return tutorial