from bs4 import BeautifulSoup

from agent.single_gen_agent.utils.state import AgentState
from agent.single_gen_agent.utils.function import safe_extract_from_soup


generator_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution. You are an expert in ComfyUI who helps users to design their own workflows.

Now you are required to create a ComfyUI workflow to finish the following task:

{query}

The key points behind the requirements and the expected paradigm of the workflow is analyzed as follows:

{analysis}

## Reference

According to the analysis, we have retrieved relevant workflows. Here are some example workflows that may be helpful:

{reference}

## Workspace

The code and description of the current workflow you are working on are presented as follows:

{workspace}

## History

Here is a recent history of your plan and action in the previous steps. This only includes several records and earlier ones are omitted. The most recent record is at the bottom of history.

{history}

## Action

Based on the current workflow in your workspace, you should first judge whether the current workflow can roughly satisfy the requirements and provide a plan to either submit or modify the workflow. Your plan should be enclosed using "<plan>" tag. For example: <plan> I will refer to example 1 and cascade it with the current workflow. I will refer to example "Pose Control" and add a new node to upscale the image resolution </plan>. After that, you should choose one of the following actions and specify your choice using "<action>" tag. For example: Therefore, the action I choose is <action> Submit </action>.

- Submit: If you think the current workflow is capable of realizing the expected effects, you can submit it without any modification and finish the task.

- Restart: You should choose one of the example workflows and start over with it, so that the current workflow will be replaced by the example workflow.

- Combine: You should choose one of the example workflows and try to merge it with the current workflow, so that their functionalities can be combined (e.g. adding a upscaling module to the text-to-video pipeline).

- Adjust: You should update the current workflow by modifying one or more parameters of the existing nodes, so that the workflow can better realize the expected effects (e.g. rewriting the prompt text).

Whichever action you choose, you should provide your Python code as in the example to formulate the workflow. You should avoid nested calls in a single code line. For example: "output_2 = node_2(input_1, node_1())" should be separated into "output_1 = node_1() and output_2 = node_2(input_1, output_1)". Make sure the updated code can be executed correctly. Your code should be enclosed using "<code>" tag. For example: <code> output_1 = node_1() </code>. Finally, you should provide a brief explanation of your modified workflow and the expected effects as in the example, which should be enclosed using "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.

Refer to the history before making a decision. As a general rule, you may choose the "Restart" action only when the history is empty (i.e. the current workflow may be unrelated to the task). You should not choose the "Adjust" action twice in a row, because modifications can be merged into a single action. You have already used {current_step} steps of action. You must choose the "Submit" action within {max_step} steps to finish the task. You are expected to submit the workflow as early as possible even if you have not used all the steps. Now, provide your plan, action, code, and description with the required format.
'''


def get_generator_agent_prompt(query: str, analysis: str, references: list, state: AgentState):
    query_text = query
    analysis_text = analysis
    reference_text = ''
    for reference in references:
        reference_text += f'- Example: {reference.metadata["name"]}\n\n'
        with open(reference.metadata['code'], 'r') as code_file:
            code = code_file.read()
        reference_text += f'<code>\n{code}\n</code>\n\n'
        with open(reference.metadata['description'], 'r') as desc_file:
            desc = desc_file.read()
        reference_text += f'<description>\n{desc}\n</description>\n\n'
    if not state.hist:
        history_text = '- The history is empty.'
    else:
        history_text = ''
        for record in state.hist[-5:]:
            plan = record['plan']
            action = record['action']
            history_text += f'- <plan> {plan} </plan> Therefore, the action I choose is <action> {action} </action>.\n\n'
    workspace_text = f'{state.code}\n\n{state.desc}'
    prompt_text = generator_prompt.format(
        query=query_text,
        analysis=analysis_text,
        reference=reference_text,
        workspace=workspace_text,
        history=history_text,
        current_step=state.step,
        max_step=state.max_step
    )
    return prompt_text


def parse_generator_agent_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    plan = safe_extract_from_soup(soup, 'plan')
    action = safe_extract_from_soup(soup, 'action')
    code = safe_extract_from_soup(soup, 'code')
    desc = safe_extract_from_soup(soup, 'description')
    return plan, action, code, desc
