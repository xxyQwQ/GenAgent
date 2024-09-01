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
