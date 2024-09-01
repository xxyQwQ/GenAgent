from bs4 import BeautifulSoup

from agent.zero_shot_agent.utils.function import safe_extract_from_soup


generator_prompt = '''
ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution. You are an expert in ComfyUI who helps users to design their own workflows.

Given a user query, you should first provide your Python code to formulate the workflow. You should avoid nested calls in a single code line. For example: "output_2 = node_2(input_1, node_1())" should be separated into "output_1 = node_1() and output_2 = node_2(input_1, output_1)". Your code should be enclosed using "<code>" tag. For example: <code> output_1 = node_1() </code>. After that, you should provide a brief description of the modules and effects of your workflow. Your description should be enclosed with "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.

Now you are required to create a ComfyUI workflow to finish the following task:

{query}
'''


def get_generator_agent_prompt(query: str):
    query_text = query
    prompt_text = generator_prompt.format(
        query=query_text
    )
    return prompt_text


def parse_generator_agent_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    description = safe_extract_from_soup(soup, 'description')
    return code, description
