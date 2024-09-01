# create nodes by instantiation
upscalemodelloader_14 = UpscaleModelLoader(model_name="""4x-UltraSharp.pth""")
reactorfaceswap_13 = ReActorFaceSwap(enabled=True, swap_model="""inswapper_128.onnx""", facedetection="""retinaface_resnet50""", face_restore_model="""codeformer-v0.1.0.pth""", face_restore_visibility=1, codeformer_weight=0.5, detect_gender_input="""no""", detect_gender_source="""no""", input_faces_index="""0""", source_faces_index="""0""", console_log_level=0)
loadimage_17 = LoadImage(image="""target.jpg""")
loadimage_18 = LoadImage(image="""source.jpg""")
imageupscalewithmodel_15 = ImageUpscaleWithModel()
saveimage_20 = SaveImage(filename_prefix="""swapped""")

# link nodes by invocation
upscale_model_14 = upscalemodelloader_14()
image_17, mask_17 = loadimage_17()
image_18, mask_18 = loadimage_18()
image_13, face_model_13 = reactorfaceswap_13(input_image=image_17, source_image=image_18, face_model=None)
image_15 = imageupscalewithmodel_15(upscale_model=upscale_model_14, image=image_13)
result_20 = saveimage_20(images=image_15)
