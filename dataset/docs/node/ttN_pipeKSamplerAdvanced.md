- `ttN pipeKSamplerAdvanced`: The `ttN pipeKSamplerAdvanced` node is designed to enhance the sampling process within a pipeline by incorporating advanced techniques and parameters. It aims to provide more control and flexibility over the generation process, allowing for customized sampling strategies that can adapt to various requirements and scenarios.
    - Parameters:
        - `lora_name`: Defines the LoRA (Low-Rank Adaptation) model name to be used, enhancing the sampling process by applying specific model adaptations. Type should be `COMBO[STRING]`.
        - `lora_model_strength`: Specifies the strength of the model adjustments made by the LoRA adaptation. Type should be `FLOAT`.
        - `lora_clip_strength`: Determines the intensity of the clip adjustments applied through LoRA adaptation. Type should be `FLOAT`.
        - `upscale_method`: Indicates the method used for upscaling images in the sampling process, affecting image quality and resolution. Type should be `COMBO[STRING]`.
        - `factor`: Defines the factor by which images are upscaled, directly influencing the output image size. Type should be `FLOAT`.
        - `crop`: Specifies whether and how the output images are cropped, affecting the final image composition. Type should be `COMBO[STRING]`.
        - `sampler_state`: unknown Type should be `COMBO[STRING]`.
        - `add_noise`: Indicates whether noise is added to the sampling process, affecting the texture and detail of generated images. Type should be `COMBO[STRING]`.
        - `steps`: unknown Type should be `INT`.
        - `cfg`: unknown Type should be `FLOAT`.
        - `sampler_name`: unknown Type should be `COMBO[STRING]`.
        - `scheduler`: unknown Type should be `COMBO[STRING]`.
        - `start_at_step`: Defines the starting step of the sampling process, allowing for control over the generation's initial state. Type should be `INT`.
        - `end_at_step`: Sets the ending step of the sampling process, determining when the generation concludes. Type should be `INT`.
        - `return_with_leftover_noise`: Specifies whether the output includes leftover noise, affecting the final image's texture and detail. Type should be `COMBO[STRING]`.
        - `image_output`: Indicates the format or destination for the generated images, affecting how and where outputs are saved. Type should be `COMBO[STRING]`.
        - `save_prefix`: Defines a prefix for saved file names, organizing outputs in a consistent manner. Type should be `STRING`.
        - `noise_seed`: Sets a seed for noise generation, ensuring reproducibility in the sampling process. Type should be `INT`.
    - Inputs:
        - `pipe`: unknown Type should be `PIPE_LINE`.
        - `optional_model`: Allows for the specification of an alternative model for sampling, providing flexibility in model usage. Type should be `MODEL`.
        - `optional_positive`: Enables the inclusion of additional positive conditioning, refining the generation towards desired attributes. Type should be `CONDITIONING`.
        - `optional_negative`: Permits the addition of negative conditioning to steer the generation away from certain attributes. Type should be `CONDITIONING`.
        - `optional_latent`: Provides an option to include a specific latent space configuration, influencing the starting point of generation. Type should be `LATENT`.
        - `optional_vae`: Allows for the use of an alternative VAE model, affecting the encoding and decoding processes. Type should be `VAE`.
        - `optional_clip`: Enables the specification of an alternative CLIP model, impacting the alignment between text and image features. Type should be `CLIP`.
        - `xyPlot`: Specifies the configuration for plotting XY data, potentially used for visualizing aspects of the sampling process. Type should be `XYPLOT`.
    - Outputs:
        - `pipe`: The modified pipeline configuration after applying advanced sampling techniques. Type should be `PIPE_LINE`.
        - `model`: The model used or modified during the advanced sampling process. Type should be `MODEL`.
        - `positive`: Positive conditioning factors applied or generated during sampling. Type should be `CONDITIONING`.
        - `negative`: Negative conditioning factors applied or generated during sampling. Type should be `CONDITIONING`.
        - `latent`: The latent space configuration resulting from the sampling process. Type should be `LATENT`.
        - `vae`: The VAE model used or modified during the sampling process. Type should be `VAE`.
        - `clip`: The CLIP model used or modified during the sampling process. Type should be `CLIP`.
        - `image`: The final image output generated by the advanced sampling process. Type should be `IMAGE`.
        - `seed`: The seed used during the sampling process, affecting reproducibility. Type should be `INT`.