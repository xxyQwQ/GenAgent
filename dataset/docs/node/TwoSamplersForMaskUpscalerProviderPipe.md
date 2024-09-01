- `TwoSamplersForMaskUpscalerProviderPipe`: This node is designed to provide a pipeline that integrates two distinct samplers specifically for the purpose of upscaling masks. It facilitates the enhancement of image quality by applying specialized sampling techniques to areas designated by masks, thereby improving the overall visual impact of the images.
    - Parameters:
        - `scale_method`: Specifies the method used for scaling during the upscaling process. It influences how the image is enlarged and the quality of the upscaling. Type should be `COMBO[STRING]`.
        - `full_sample_schedule`: Defines the schedule for sampling throughout the upscaling process. It determines the sequence and parameters for sampling operations. Type should be `COMBO[STRING]`.
        - `use_tiled_vae`: Indicates whether a tiled VAE approach is used for upscaling. This affects the handling of large images by breaking them into tiles for processing. Type should be `BOOLEAN`.
        - `tile_size`: The size of the tiles used when a tiled VAE approach is employed. It specifies the dimensions for breaking down large images. Type should be `INT`.
    - Inputs:
        - `base_sampler`: Specifies the base sampler used in the upscaling process, which is crucial for the initial sampling phase. Type should be `KSAMPLER`.
        - `mask_sampler`: Defines the sampler used specifically for the mask areas during upscaling, enhancing the details in these regions. Type should be `KSAMPLER`.
        - `mask`: The mask that designates areas for specialized upscaling, playing a key role in the targeted enhancement of image quality. Type should be `MASK`.
        - `basic_pipe`: The basic processing pipeline that provides essential functionalities like VAE operations. It serves as the foundation for the upscaling process. Type should be `BASIC_PIPE`.
        - `full_sampler_opt`: Optional configurations for the full sampler used in the upscaling process. It allows customization of the sampling behavior. Type should be `KSAMPLER`.
        - `upscale_model_opt`: Optional configurations for the upscale model. It enables fine-tuning of the model's parameters for better upscaling results. Type should be `UPSCALE_MODEL`.
        - `pk_hook_base_opt`: Optional configurations for the base hook in the pipeline. It affects the initial phase of the upscaling process. Type should be `PK_HOOK`.
        - `pk_hook_mask_opt`: Optional configurations for the mask hook. It influences how the mask is applied and processed during upscaling. Type should be `PK_HOOK`.
        - `pk_hook_full_opt`: Optional configurations for the full hook, affecting the entire upscaling process. It allows for comprehensive customization of the upscaling behavior. Type should be `PK_HOOK`.
    - Outputs:
        - `upscaler`: The result of the upscaling process, providing an enhanced version of the image with improved quality in masked areas. Type should be `UPSCALER`.