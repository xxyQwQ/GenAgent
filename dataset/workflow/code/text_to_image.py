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
