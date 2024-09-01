import json


class AgentState(object):
    def __init__(self, max_step=5):
        self.max_step = max_step
        with open('./dataset/workflow/meta.json') as meta_file:
            metadata = json.load(meta_file)
        path = metadata['text_to_image']
        with open(path['code'], 'r') as code_file:
            self.code = code_file.read()
        with open(path['description'], 'r') as desc_file:
            self.desc = desc_file.read()
        self.step = 0
        self.hist = []

    def update(self, plan, action, code, desc):
        self.code = code
        self.desc = desc
        self.step += 1
        self.hist.append({
            'plan': plan,
            'action': action,
            'code': code,
            'desc': desc
        })
