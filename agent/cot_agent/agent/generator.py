from bs4 import BeautifulSoup

from agent.cot_agent.utils.function import safe_extract_from_soup


generator_prompt = '''
ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution. You are an expert in ComfyUI who helps users to design their own workflows.

Here is an example workflow for your reference:

<code>
# create nodes by instantiation
vaedecode_8 = VAEDecode()
saveimage_9 = SaveImage(filename_prefix="""ComfyUI""")
cliptextencode_6 = CLIPTextEncode(text="""a photo of a cat wearing a spacesuit inside a spaceship  high resolution, detailed, 4k""")
cliptextencode_7 = CLIPTextEncode(text="""blurry, illustration""")
emptylatentimage_5 = EmptyLatentImage(width=512, height=512, batch_size=1)
ksampler_3 = KSampler(seed=636250194499614, control_after_generate="""fixed""", steps=20, cfg=7, sampler_name="""dpmpp_2m""", scheduler="""karras""", denoise=1)
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""dreamshaper_8.safetensors""")

# link nodes by invocation
latent_5 = emptylatentimage_5()
model_4, clip_4, vae_4 = checkpointloadersimple_4()
conditioning_6 = cliptextencode_6(clip=clip_4)
conditioning_7 = cliptextencode_7(clip=clip_4)
latent_3 = ksampler_3(model=model_4, positive=conditioning_6, negative=conditioning_7, latent_image=latent_5)
image_8 = vaedecode_8(samples=latent_3, vae=vae_4)
result_9 = saveimage_9(images=image_8)
</code>

This workflow presents a basic pipeline of text-to-image generation using Stable Diffusion. Given the positive and negative prompts, the workflow will load the pretrained models, encode the prompts into conditioning, create and denoise the latent code, and finally decode the latent code into an image. In this example, we generate a photo of a cat wearing a spacesuit inside a spaceship, while avoiding blurry and illustration-like results.

Now you are required to create a ComfyUI workflow to finish the following task:

{query}

First, you should provide your step-by-step plan, including which nodes you will use and how you will link them. Your plan should be enclosed using "<plan>" tag. For example: <plan> Step 1: I will use the EmptyLatentImage node to create an empty latent image. Step 2: I will use the KSampler node to denoise the latent image. </plan>.

After that, you should write the Python code following your plan to formulate the workflow. You should avoid nested calls in a single code line. For example: "output_2 = node_2(input_1, node_1())" should be separated into "output_1 = node_1() and output_2 = node_2(input_1, output_1)". Your code should be enclosed using "<code>" tag. For example: <code> output_1 = node_1() </code>.
'''


def get_generator_agent_prompt(query: str):
    query_text = query
    prompt_text = generator_prompt.format(
        query=query_text
    )
    return prompt_text


def parse_generator_agent_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    plan = safe_extract_from_soup(soup, 'plan')
    code = safe_extract_from_soup(soup, 'code')
    return plan, code
