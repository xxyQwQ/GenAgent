from bs4 import BeautifulSoup

from agent.few_shot_agent.utils.function import safe_extract_from_soup


generator_prompt = '''
ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution. You are an expert in ComfyUI who helps users to design their own workflows.

Given a user query, you should first provide your Python code to formulate the workflow. You should avoid nested calls in a single code line. For example: "output_2 = node_2(input_1, node_1())" should be separated into "output_1 = node_1() and output_2 = node_2(input_1, output_1)". Your code should be enclosed using "<code>" tag. For example: <code> output_1 = node_1() </code>. After that, you should provide a brief description of the modules and effects of your workflow. Your description should be enclosed with "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.

Here are some example tasks and corresponding answers for your reference:

- Generate a photo of a cat wearing a spacesuit inside a spaceship, while avoiding blurry and illustration-like results.

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

<description>
This workflow presents a basic pipeline of text-to-image generation using Stable Diffusion. Given the positive and negative prompts, the workflow will load the pretrained models, encode the prompts into conditioning, create and denoise the latent code, and finally decode the latent code into an image, thus generating a photo of a cat wearing a spacesuit inside a spaceship without blurry and illustration.
</description>

- Upscale the image `titled_book.png` by 2x while maintaining the quality.

<code>
# create nodes by instantiation
imageupscalewithmodel_12 = ImageUpscaleWithModel()
imagescaleby_30 = ImageScaleBy(upscale_method="""bilinear""", scale_by=0.5)
upscalemodelloader_11 = UpscaleModelLoader(model_name="""4x-UltraSharp.pth""")
loadimage_31 = LoadImage(image="""titled_book.png""")
saveimage_29 = SaveImage(filename_prefix="""ComfyUI""")

# link nodes by invocation
upscale_model_11 = upscalemodelloader_11()
image_31, mask_31 = loadimage_31()
image_12 = imageupscalewithmodel_12(upscale_model=upscale_model_11, image=image_31)
image_30 = imagescaleby_30(image=image_12)
result_29 = saveimage_29(images=image_30)
</code>

<description>
This workflow uses Upscale Model to upscale images. Given an image, the workflow will upscale it into high resolution. Since the model will upscale the image by 4x, we add another node for 0.5x downscale.
</description>

- The video `play_guitar.gif` contains 8 frames per second. Interpolate the frames to 24 frames per second.

<code>
# create nodes by instantiation
vhs_videocombine_3 = VHS_VideoCombine(frame_rate=24, loop_count=0, filename_prefix="""AnimateDiff""", format="""image/gif""", pingpong=False, save_output=True)
vhs_loadvideo_7 = VHS_LoadVideo(video="""play_guitar.gif""", force_rate=0, force_size="""Disabled""", custom_width=512, custom_height=512, frame_load_cap=0, skip_first_frames=0, select_every_nth=1)
rife_vfi_10 = RIFE_VFI(ckpt_name="""rife47.pth""", clear_cache_after_n_frames=10, multiplier=3, fast_mode=True, ensemble=True, scale_factor=1)

# link nodes by invocation
image_7, frame_count_7, audio_7, video_info_7 = vhs_loadvideo_7(meta_batch=None)
image_10 = rife_vfi_10(frames=image_7, optional_interpolation_states=None)
filenames_3 = vhs_videocombine_3(images=image_10, audio=None, meta_batch=None)
</code>

<description>
This workflow uses the RIFE VFI node to interpolate frames. Given the input video, the workflow will interpolate the frames with a certain multiplier and combine them into a new video. We interpolate the video by 3x, so the output video will have 24 frames per second.
</description>

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
