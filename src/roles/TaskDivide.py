

from metagpt.roles.role import Role
from metagpt.schema import Message
from metagpt.actions import UserRequirement

from metagpt.logs import logger
from actions.TaskDivideAction import TaskDivideAction
import sys


class TaskDivide(Role):
    name: str = "taskdivide"
    profile: str = "TaskDivide"

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        self.set_actions([TaskDivideAction]) 

        # 订阅消息
        self._watch({UserRequirement}) 


    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        user_re = self.rc.history[0]
        print('self.rc.history:',user_re.content)
        CFD_task, CFD_post_task, CFD_analysis_task, CFD_optimization_task= await todo.run(self.rc.history)  #根据历史输入，找到任务分解的结果 

        self.rc.env.publish_message(Message(content=user_re.content, cause_by=TaskDivideAction))

        return Message(content=CFD_task, role=self.profile, cause_by=type(todo)) 
    