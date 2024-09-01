from bs4 import BeautifulSoup

from agent.json_gen_agent.utils.state import AgentState
from agent.json_gen_agent.utils.function import safe_extract_from_soup


combiner_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Workflows are saved in JSON format, describing the attributes of nodes and links.
You are an expert in ComfyUI who helps users to design their own workflows.

Now you are required to create a ComfyUI workflow to finish the following task:

{query}

The key points behind the requirements and the expected paradigm of the workflow is analyzed as follows:

{analysis}

## Reference

The json and description of the example workflow you are referring to are presented as follows:

{reference}

## Workspace

The json and description of the current workflow you are working on are presented as follows:

{workspace}

## Combination

Based on the current working progress, your step-by-step plan is presented as follows:

{planning}

Now you are working on the first step of your plan. Later steps should not be considered at this moment.
In other words, you should merge the example workflow with the current workflow according to your plan, so that their functions can be combined.

First, you should provide your json to formulate the updated workflow.
Your json should be enclosed with "<json>" tag. For example: <json> [1, 2, 3] </json>.

After that, you should provide a brief description of the updated workflow and the expected effects as in the example.
Your description should be enclosed with "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.

Now, provide your json and description with the required format.
'''


def get_combiner_agent_prompt(state: AgentState, planning: str, reference: dict):
    query_content = state.query
    analysis_content = state.analysis

    reference_content = f'- Example: {reference.metadata["name"]}\n\n'
    with open(reference.metadata['workflow'], 'r') as json_file:
        workflow = json_file.read()
    reference_content += f'<json>\n{workflow}\n</json>\n\n'
    with open(reference.metadata['description'], 'r') as desc_file:
        description = desc_file.read()
    reference_content += f'<description>\n{description}\n</description>'

    workspace_content = f'<json>\n{state.workspace["workflow"]}\n</json>\n\n'
    workspace_content += f'<description>\n{state.workspace["description"]}\n</description>'
    planning_content = planning

    prompt_text = combiner_prompt.format(
        query=query_content,
        analysis=analysis_content,
        reference=reference_content,
        workspace=workspace_content,
        planning=planning_content
    )
    return prompt_text


def parse_combiner_agent_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    workflow = safe_extract_from_soup(soup, 'json')
    description = safe_extract_from_soup(soup, 'description')
    return workflow, description
