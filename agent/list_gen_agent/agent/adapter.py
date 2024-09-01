from bs4 import BeautifulSoup

from agent.list_gen_agent.utils.state import AgentState
from agent.list_gen_agent.utils.function import safe_extract_from_soup


adapter_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Workflows are described by unordered lists in Markdown format, including the nodes, links and parameters.
You are an expert in ComfyUI who helps users to design their own workflows.

Now you are required to create a ComfyUI workflow to finish the following task:

{query}

The key points behind the requirements and the expected paradigm of the workflow is analyzed as follows:

{analysis}

## Workspace

The markdown and description of the current workflow you are working on are presented as follows:

{workspace}

## Adaptation

Based on the current working progress, your step-by-step plan is presented as follows:

{planning}

Now you are working on the first step of your plan. Later steps should not be considered at this moment.
In other words, you should adjust some of the parameters in the current workflow according to your plan, so that you can better realize the expected effects. Your expected modification is specified as follows:

{adaptation}

First, you should provide your markdown to formulate the updated workflow.
Your markdown should be enclosed with "<markdown>" tag. For example: <markdown> - node_type: "SaveImage" </markdown>.

After that, you should provide a brief description of the updated workflow and the expected effects as in the example.
Your description should be enclosed with "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.

Now, provide your markdown and description with the required format.
'''


def get_adapter_agent_prompt(state: AgentState, planning: str, adaptation: str):
    query_content = state.query
    analysis_content = state.analysis
    workspace_content = f'<markdown>\n{state.workspace["markdown"]}\n</markdown>\n\n'
    workspace_content += f'<description>\n{state.workspace["description"]}\n</description>'
    planning_content = planning
    adaptation_content = adaptation

    prompt_text = adapter_prompt.format(
        query=query_content,
        analysis=analysis_content,
        workspace=workspace_content,
        planning=planning_content,
        adaptation=adaptation_content
    )
    return prompt_text


def parse_adapter_agent_response(response):
    soup = BeautifulSoup(response, 'html.parser')
    markdown = safe_extract_from_soup(soup, 'markdown')
    description = safe_extract_from_soup(soup, 'description')
    return markdown, description
