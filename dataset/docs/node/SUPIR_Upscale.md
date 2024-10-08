- `SUPIR_Upscale`: The SUPIR_Upscale node is designed to upscale images using the SUPIR model, enhancing image resolution while maintaining or improving image quality. This node is part of a suite of nodes aimed at image processing and enhancement, leveraging advanced deep learning techniques to achieve superior upscaling results.
    - Parameters:
        - `supir_model`: Specifies the SUPIR model to be used for upscaling, allowing for customization of the upscaling process. Type should be `COMBO[STRING]`.
        - `sdxl_model`: Determines the secondary model used in conjunction with the SUPIR model to enhance the upscaling process. Type should be `COMBO[STRING]`.
        - `seed`: Sets the seed for random number generation, ensuring reproducibility of the upscaling results. Type should be `INT`.
        - `resize_method`: Determines the method used for resizing images during the upscaling process, affecting the texture and quality of the output. Type should be `COMBO[STRING]`.
        - `scale_by`: Specifies the factor by which the image will be upscaled, directly influencing the resolution of the output image. Type should be `FLOAT`.
        - `steps`: Defines the number of steps to be used in the upscaling process, impacting the detail and quality of the upscaled image. Type should be `INT`.
        - `restoration_scale`: Adjusts the scale of restoration applied to the upscaled image, affecting the balance between detail enhancement and artifact reduction. Type should be `FLOAT`.
        - `cfg_scale`: Controls the configuration scale for the upscaling process, influencing the adherence to the input image's content and style. Type should be `FLOAT`.
        - `a_prompt`: Provides a positive textual description to guide the upscaling process, enhancing certain aspects of the image according to the specified attributes. Type should be `STRING`.
        - `n_prompt`: Provides a negative textual description to avoid certain aspects during the upscaling process, helping to steer the result away from undesired attributes. Type should be `STRING`.
        - `s_churn`: Specifies the churn rate in the sampling process, affecting the exploration of the latent space and the diversity of the upscaled results. Type should be `INT`.
        - `s_noise`: Sets the noise level in the sampling process, influencing the amount of randomness and potentially the detail in the upscaled image. Type should be `FLOAT`.
        - `control_scale`: Adjusts the control scale for the upscaling process, affecting the overall control over the upscaling outcome. Type should be `FLOAT`.
        - `cfg_scale_start`: Specifies the starting configuration scale, allowing for dynamic adjustment of the cfg scale throughout the upscaling process. Type should be `FLOAT`.
        - `control_scale_start`: Specifies the starting control scale, enabling dynamic adjustment of control over the upscaling outcome throughout the process. Type should be `FLOAT`.
        - `color_fix_type`: unknown Type should be `COMBO[STRING]`.
        - `keep_model_loaded`: unknown Type should be `BOOLEAN`.
        - `use_tiled_vae`: unknown Type should be `BOOLEAN`.
        - `encoder_tile_size_pixels`: unknown Type should be `INT`.
        - `decoder_tile_size_latent`: unknown Type should be `INT`.
        - `captions`: unknown Type should be `STRING`.
        - `diffusion_dtype`: unknown Type should be `COMBO[STRING]`.
        - `encoder_dtype`: unknown Type should be `COMBO[STRING]`.
        - `batch_size`: unknown Type should be `INT`.
        - `use_tiled_sampling`: unknown Type should be `BOOLEAN`.
        - `sampler_tile_size`: unknown Type should be `INT`.
        - `sampler_tile_stride`: unknown Type should be `INT`.
        - `fp8_unet`: unknown Type should be `BOOLEAN`.
        - `fp8_vae`: unknown Type should be `BOOLEAN`.
        - `sampler`: unknown Type should be `COMBO[STRING]`.
    - Inputs:
        - `image`: Specifies the image to be upscaled, serving as the primary input for the upscaling process. Type should be `IMAGE`.
    - Outputs:
        - `upscaled_image`: The result of the upscaling process, showcasing enhanced resolution and quality. Type should be `IMAGE`.
