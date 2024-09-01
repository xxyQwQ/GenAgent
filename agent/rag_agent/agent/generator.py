from bs4 import BeautifulSoup

from agent.rag_agent.utils.function import safe_extract_from_soup


generator_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution. You are an expert in ComfyUI who helps users to design their own workflows.

Now you are required to create a ComfyUI workflow to finish the following task:

{query}

## Reference

According to the requirements, we have retrieved relevant workflows. Here are some example workflows that may be helpful:

{reference}

## Format

First, you should provide your step-by-step plan, including which workflows you will refer to and how you will modify and compose them. Your plan should be enclosed using "<plan>" tag. For example: <plan> I will create my workflow based on example 1 and 2. I will cascade them and rewrite the prompt text. In addition, I will add a new node to upscale the image resolution. </plan>.

After that, you should provide your Python code as in the example to formulate the workflow. You should avoid nested calls in a single code line. For example: "output_2 = node_2(input_1, node_1())" should be separated into "output_1 = node_1() and output_2 = node_2(input_1, output_1)". Your code should be enclosed using "<code>" tag. For example: <code> output_1 = node_1() </code>.
'''


def get_generator_agent_prompt(query: str, references: list):
    query_text = query
    reference_text = ''
    for reference in references:
        reference_text += f'- Example: {reference.metadata["name"]}\n\n'
        with open(reference.metadata['code'], 'r') as code_file:
            code = code_file.read()
        reference_text += f'<code>\n{code}\n</code>\n\n'
        with open(reference.metadata['description'], 'r') as desc_file:
            desc = desc_file.read()
        reference_text += f'<description>\n{desc}\n</description>\n\n'
    prompt_text = generator_prompt.format(
        query=query_text,
        reference=reference_text
    )
    return prompt_text


def parse_generator_agent_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    plan = safe_extract_from_soup(soup, 'plan')
    code = safe_extract_from_soup(soup, 'code')
    return plan, code
