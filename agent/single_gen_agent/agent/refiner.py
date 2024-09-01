from bs4 import BeautifulSoup

from agent.single_gen_agent.utils.state import AgentState
from agent.single_gen_agent.utils.function import safe_extract_from_soup


refiner_prompt = '''
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

The code and description of the current workflow you are working on is presented as follows:

{workspace}

## Troubleshooting

In the last step, your plan to realize the expected effects is presented as follows:

{plan}

The latest code and description of your modified workflow are presented as follows:

{traceback}

However, an error occurred when converting your code into a workflow. This may be caused by nested calls, missing parameters, or other issues. The detailed error message is presented as follows:

{error}

Try to explain why the error occurred, which should be enclosed using "<explanation>" tag. For example: <explanation> The error occurred because the input parameter of node_1 is missing. </explanation>.

After that, correct the error and provide your Python code again. You should avoid nested calls in a single code line. For example: "output_2 = node_2(input_1, node_1())" should be separated into "output_1 = node_1() and output_2 = node_2(input_1, output_1)". Your code should be enclosed using "<code>" tag. For example: <code> output_1 = node_1() </code>.

Finally, provide a brief explanation of your modified workflow and the expected effects as in the example, which should be enclosed using "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.
'''


def get_refiner_agent_prompt(query: str, analysis: str, references: list, state: AgentState, plan: str, code: str, desc: str, error: str):
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
    workspace_text = f'{state.code}\n\n{state.desc}'
    plan_text = plan
    traceback_text = f'{code}\n\n{desc}'
    error_text = error
    prompt_text = refiner_prompt.format(
        query=query_text,
        analysis=analysis_text,
        reference=reference_text,
        workspace=workspace_text,
        plan=plan_text,
        traceback=traceback_text,
        error=error_text
    )
    return prompt_text


def parse_refiner_agent_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    expl = safe_extract_from_soup(soup, 'explanation')
    code = safe_extract_from_soup(soup, 'code')
    desc = safe_extract_from_soup(soup, 'description')
    return expl, code, desc
