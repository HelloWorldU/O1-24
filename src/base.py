import os
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
model_path = r"C:\Users\Administrator\Desktop\LLM\O1-24\models\Qwen2.5-0.5b-Instruct\cache\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775"

class Task:
    def __init__(self):
        pass

    def __len__(self) -> int:
        pass

    def get_input(self, idx: int) -> str:
        pass


def get_task(name):
    if name == 'game24':
        from tasks.game24 import Game24Task
        return Game24Task()
    # elif name == 'text':
    #     from tot.tasks.text import TextTask
    #     return TextTask()
    # elif name == 'crosswords':
    #     from tot.tasks.crosswords import MiniCrosswordsTask
    #     return MiniCrosswordsTask()
    else:
        raise NotImplementedError