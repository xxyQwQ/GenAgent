class AgentState(object):
    def __init__(self, step: int, query: str, analysis: str, reference: list = None, workspace: dict = None):
        self.step = step
        self.query = query
        self.analysis = analysis
        self.reference = reference
        self.workspace = workspace
        self.history = []

    def update_reference(self, reference: list):
        self.reference = reference

    def update_workspace(self, code: str, description: str):
        self.workspace['code'] = code
        self.workspace['description'] = description

    def update_history(self, thought: str, plan: str, action: str):
        self.history.append({
            'step': self.step,
            'thought': thought,
            'plan': plan,
            'action': action
        })
        self.step -= 1
