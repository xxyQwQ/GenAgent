import ast
from bs4 import BeautifulSoup

from agent.code_gen_agent.utils.state import AgentState
from agent.code_gen_agent.utils.function import safe_extract_from_soup


planner_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution.
You are an expert in ComfyUI who helps users to design their own workflows.

Now you are required to create a ComfyUI workflow to finish the following task:

{query}

The key points behind the requirements and the expected paradigm of the workflow is analyzed as follows:

{analysis}

## Reference

According to the analysis, we have retrieved relevant workflows. Here are some example workflows that may be helpful:

{reference}

## History

Here is a recent history of your thought, plan and action in the previous steps. This only includes several records and earlier ones are omitted. The most recent record is at the bottom of history.

{history}

## Workspace

The code and description of the current workflow you are working on are presented as follows:

{workspace}

## Action

Based on your previous plan and the current workflow, you should first think about what functions have been implemented and what modules remain to be added.
Your thought should be enclosed with "<thought>" tag. For example: <thought> The text-to-image pipeline has been implemented, but an upscaling module is needed to improve the resolution. </thought>.

After that, you should update your step-by-step plan to further modify your workflow.
Your plan should contain at most {step} steps, but fewer steps will be better. Make sure that each step is feasible to be converted into a single action.
Your plan should be enclosed with "<plan>" tag. For example: <plan> Step 1: I will refer to example_name to add a new node. Step 2: I will finish the task since all the functions are implemented. </plan>.

Finally, you should choose one of the following actions and specify the arguments (if required), so that the updated workflow can realize the first step in your new plan.
You should provide your action with the format of function calls in Python.
Your action should be enclosed with "<action>" tag. For example: <action> combine(name="example_name") </action>, <action> adapt(prompt="Change the factor to 0.5 and rewrite the prompt.") </action>, and <action> finish() </action>.

- `load`: Load the specified example workflow into your workspace to replace the current workflow, so that you can start over. Arguments:
  - `name`: The name of the example workflow you want to load.
- `combine`: Combine the current workflow with the specified example workflow, so that you can add necessary modules (e.g. adding a upscaling module to the text-to-image pipeline). Arguments:
  - `name`: The name of the example workflow you want to combine.
- `adapt`: Adapt some of the parameters in the current workflow, so that you can better realize the expected effects. Arguments:
  - `prompt`: The prompt to specify which parameters you want to adapt and how to adapt them.
- `retrieve`: Retrieve a new batch of example workflows, so that you may find useful references. Arguments:
  - `prompt`: The prompt to describe the expected features of example workflows you want to retrieve.
- `finish`: Finish the task since the current workflow is capable of realizing the expected effects.

Refer to the history before making a decision. Here are some general rules you should follow:

1. You may choose the `load` action only when the history is empty (i.e. the current workflow may be unrelated to the task).
2. If you choose the `load` or `combine` action, make sure that the example workflow exists in the reference. Otherwise, you should try to update the reference list by the `retrieve` action.
3. You should not choose the `adapt` action twice in a row, because they can be simplified into a single action.
4. If you choose the `adapt` or `retrieve` action, make sure that your prompt is concise and contains all the necessary information.
5. You should choose the `finish` action before the steps are exhausted.

Now, provide your thought, plan and action with the required format.
'''


def get_planner_agent_prompt(state: AgentState):
    query_content = state.query
    analysis_content = state.analysis

    reference_content = ''
    for reference in state.reference:
        reference_content += f'- Example: {reference.metadata["name"]}\n\n'
        with open(reference.metadata['description'], 'r') as desc_file:
            description = desc_file.read()
        reference_content += f'<description>\n{description}\n</description>\n\n'

    history_content = ''
    if not state.history:
        history_content = '- The history is empty.'
    else:
        for record in state.history[-5:]:
            history_content += f'<plan> {record["plan"]} </plan>\n'
            history_content += f'<action> {record["action"]} </action>.\n\n'

    workspace_content = f'<code>\n{state.workspace["code"]}\n</code>\n\n'
    workspace_content += f'<description>\n{state.workspace["description"]}\n</description>'

    prompt_text = planner_prompt.format(
        query=query_content,
        analysis=analysis_content,
        reference=reference_content,
        history=history_content,
        workspace=workspace_content,
        step=state.step
    )
    return prompt_text


def parse_planner_agent_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    thought = safe_extract_from_soup(soup, 'thought')
    plan = safe_extract_from_soup(soup, 'plan')
    action = safe_extract_from_soup(soup, 'action')
    return thought, plan, action


def parse_planner_agent_action(action: str):
    node = ast.parse(action)
    call = node.body[0].value
    command = call.func.id
    arguments = {}
    for keyword in call.keywords:
        arguments[keyword.arg] = keyword.value.value
    return command, arguments
