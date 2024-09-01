analyzer_prompt = '''
ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution.
You are an expert in ComfyUI who helps users to design their own workflows.

Now you are required to create a ComfyUI workflow to finish the following task:

{query}

Based on the description, point out the key points behind the requirements (e.g. main object, specific style, target resolution, etc.) and the expected paradigm of the workflow (e.g., text-to-image, image-to-image, image-to-video, etc.).
You are not required to provide the code for the workflow. Please make sure your answers are clear and concise within a single paragraph.
'''


def get_analyzer_agent_prompt(query: str):
    query_content = query
    prompt_text = analyzer_prompt.format(
        query=query_content
    )
    return prompt_text
