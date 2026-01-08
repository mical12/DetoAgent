
import asyncio

from metagpt.context import Context
from metagpt.schema import Message
from metagpt.environment.base_env import Environment

from qa_module import AsyncQA_tutorial, AsyncQA_tutorial_name, AsyncQA_allrun, AsyncQA_Ori, AsyncImageQA
from roles.Architect import Architect
from roles.InputWriter import InputWriter
from roles.Reviewer import Reviewer
from roles.Runner import Runner
from roles.TaskDivide import TaskDivide
from langchain_huggingface import HuggingFaceEmbeddings
import config_path
from Statistics import Statistics, global_statistics
import time
import logging, os, datetime

async def main():
    overall_stats = Statistics()         #创建统计时间的类
    for _ in range(config_path.run_times):   #_表示不用变量名，匿名表示
        global_statistics.reset()            #准备新一轮统计
        start_time = time.time()
        global_statistics.runtimes = _ + 1
        print("runtimes:",_ + 1)
        config_path.should_stop = False
        await run_instance()

        global_statistics.running_time = time.time() - start_time
        global_statistics.save_to_file(config_path.Case_PATH)
        overall_stats.save(global_statistics)

    overall_stats.average(config_path.run_times)
    overall_stats.display()
    overall_stats.save_ave_file(config_path.Case_PATH)
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = f"logs/run_{timestamp}.log"

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=log_path,          # ✨ 保存到文件
        filemode="w"
    )

async def run_instance():
    Architect_run = 0
    embedding_model = HuggingFaceEmbeddings(
    model_name="/mnt/d/Ubantu_run/MetaOpenFOAM/Qwen3-Embedding-0.6B"
    )  # 使用HuggingFace的预训练模型进行嵌入
    async_qa = AsyncQA_tutorial()                  # 回答问题的模块
    async_qa.init_instance(embedding_model)
    async_qa_name = AsyncQA_tutorial_name()
    async_qa_name.init_instance(embedding_model)
    async_qa_allrun = AsyncQA_allrun(embedding_model)
    async_qa_allrun.init_instance(embedding_model)
    async_qa_ori = AsyncQA_Ori()
    async_qa_ori.init_instance()
    async_qa_image = AsyncImageQA()
    async_qa_image.init_instance()
    context = Context() # Load config2.yaml
    env = Environment(context=context)
    #postprocessing = PostProcessing(context=context)
    taskdivide = TaskDivide(context=context)
    architect = Architect(context=context)
    inputWriter = InputWriter(context=context)
    runner = Runner(context=context)
    reviewer = Reviewer(context=context)

    env.add_roles([taskdivide, architect, inputWriter,runner, reviewer])

    #env.publish_message(Message(content='do a DNS simulation of homogeneous isotropic turbulence with Grid N= 16^3', send_to=AgentA)) # 将用户的消息发送个Agent A，让Agent A开始工作。
    #do a DNS simulation of incompressible forcing homogeneous isotropic turbulence with Grid 16^3 using dnsFoam (success) loop 2
    #do a LES simulation (not parallel) of incompressible channel flow with Grid 50*10*10 using pimpleFoam, inlet velocity 1 m/s, length 5 cm, case name: incompressibleChannelFlow3 (success) loop 6
    #do a DNS simulation of incompressible forcing homogeneous isotropic turbulence with Grid 32^3 using dnsFoam (success)
    #do a RANS simulation of backward facing step with inlet velocity 1 m/s, without parallel (success) loop 12
    #do a simulation of flow around a cylinder with inlet velocity 1 m/s without parallel, case name: cylinder3
    #do a LES simulation (not parallel) of incompressible channel flow with Grid 50*10*10 using pimpleFoam, inlet velocity 1 m/s, length 5 cm and the blockMeshDict is already prepared.
    #do an incompressible lid driven cavity flow simulation with the top wall moves in the x direction at a speed of 1 m/s while the other 3 are stationary (no loop) (success)
    #do a LES simulation of compressible pitzDaily using rhoPimpleFoam, and the velocity of inlet is 100 m/s.
    #do a laminar simulation of incompressible planar Poiseuille flow of a non-Newtonian fluid with grid 1*20*1, modelled using the Maxwell viscoelastic laminar stress model, initially at rest, constant pressure gradient applied from time zero
    #do a compressible simulation of squareBendLiq of using rhoSimpleFoam with endTime 100 and writeInterval 10 (success)
    #do a RANS simulation of buoyantCavity using buoyantSimpleFoam, which investigate natural convection in a heat cavity with a temperature difference of 20K is maintained between the hot and cold; the remaining patches are treated as adiabatic (ex: 2)
    env.publish_message(Message(content=config_path.usr_requirment, send_to=TaskDivide))
    while not env.is_idle and not config_path.should_stop: 
        await env.run()
        

if __name__ == "__main__":
    asyncio.run(main())

