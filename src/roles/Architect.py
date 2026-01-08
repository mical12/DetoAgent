

from metagpt.roles.role import Role
from metagpt.schema import Message
from metagpt.actions import UserRequirement

from metagpt.logs import logger
from actions.ArchitectAction import ArchitectAction
from actions.TaskDivideAction import TaskDivideAction

class Architect(Role):
    name: str = "Zhuxu"
    profile: str = "Architect"

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

        self.set_actions([ArchitectAction]) 

        # 订阅消息
        self._watch({TaskDivideAction}) 


    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        user_re = self.rc.history[0]
        print('self.rc.history[0]:',user_re.content)
        print('self.rc.history:',self.rc.history)
        print('cfd task',self.rc.history[-1])

        cfd_task = self.rc.history[-1]
        
        subtasks = await todo.run(self.rc.history[-1])

        self.rc.env.publish_message(Message(content=cfd_task.content, cause_by=ArchitectAction))

        for i in subtasks:
           self.rc.env.publish_message(Message(content=i, cause_by=ArchitectAction)) 
        
        return Message(content=str(len(subtasks)), role=self.profile, cause_by=type(todo)) 
    

