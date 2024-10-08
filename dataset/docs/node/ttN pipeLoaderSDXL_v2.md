- `ttN pipeLoaderSDXL_v2`: This node is designed to load and initialize the Stable Diffusion XL model for image generation tasks, providing an enhanced version with optimizations for larger scale operations.
    - Parameters:
        - `ckpt_name`: Specifies the checkpoint name for the Stable Diffusion XL model, crucial for loading the correct model version for image generation. Type should be `COMBO[STRING]`.
        - `config_name`: Defines the configuration name, essential for setting up the model with the appropriate parameters and optimizations. Type should be `COMBO[STRING]`.
        - `vae_name`: Names the VAE used in conjunction with the Stable Diffusion XL model, important for the image generation process and quality. Type should be `COMBO[STRING]`.
        - `clip_skip`: Determines the number of layers to skip in the CLIP model, affecting the integration of textual guidance. Type should be `INT`.
        - `loras`: Specifies LoRA modifications to apply, enhancing model capabilities with additional parameters. Type should be `STRING`.
        - `refiner_ckpt_name`: Names the checkpoint for the refiner model, used to refine or alter the generated images. Type should be `COMBO[STRING]`.
        - `refiner_config_name`: unknown Type should be `COMBO[STRING]`.
        - `positive_g`: unknown Type should be `STRING`.
        - `positive_l`: unknown Type should be `STRING`.
        - `negative_g`: unknown Type should be `STRING`.
        - `negative_l`: unknown Type should be `STRING`.
        - `conditioning_aspect`: unknown Type should be `COMBO[STRING]`.
        - `conditioning_width`: unknown Type should be `INT`.
        - `conditioning_height`: unknown Type should be `INT`.
        - `crop_width`: unknown Type should be `INT`.
        - `crop_height`: unknown Type should be `INT`.
        - `target_aspect`: unknown Type should be `COMBO[STRING]`.
        - `target_width`: unknown Type should be `INT`.
        - `target_height`: unknown Type should be `INT`.
        - `positive_ascore`: unknown Type should be `INT`.
        - `negative_ascore`: unknown Type should be `INT`.
        - `empty_latent_aspect`: unknown Type should be `COMBO[STRING]`.
        - `empty_latent_width`: unknown Type should be `INT`.
        - `empty_latent_height`: unknown Type should be `INT`.
        - `seed`: unknown Type should be `INT`.
        - `prepend_positive_g`: unknown Type should be `STRING`.
        - `prepend_positive_l`: unknown Type should be `STRING`.
        - `prepend_negative_g`: unknown Type should be `STRING`.
        - `prepend_negative_l`: unknown Type should be `STRING`.
    - Inputs:
        - `model_override`: unknown Type should be `MODEL`.
        - `clip_override`: unknown Type should be `CLIP`.
        - `optional_lora_stack`: unknown Type should be `LORA_STACK`.
        - `optional_controlnet_stack`: unknown Type should be `CONTROLNET_STACK`.
        - `refiner_model_override`: unknown Type should be `MODEL`.
        - `refiner_clip_override`: unknown Type should be `CLIP`.
    - Outputs:
        - `sdxl_pipe`: Provides the initialized Stable Diffusion XL pipeline, ready for image generation tasks. Type should be `PIPE_LINE_SDXL`.
        - `model`: Returns the loaded model component of the Stable Diffusion XL pipeline. Type should be `MODEL`.
        - `positive`: Outputs the positive conditioning component, guiding the image generation towards desired themes. Type should be `CONDITIONING`.
        - `negative`: Outputs the negative conditioning component, steering the image generation away from undesired themes. Type should be `CONDITIONING`.
        - `vae`: Returns the loaded VAE component, crucial for the image encoding and decoding processes. Type should be `VAE`.
        - `clip`: Outputs the loaded CLIP model component, used for textual guidance in image generation. Type should be `CLIP`.
        - `refiner_model`: Provides the loaded refiner model component, used for refining or altering the generated images. Type should be `MODEL`.
        - `refiner_positive`: unknown Type should be `CONDITIONING`.
        - `refiner_negative`: unknown Type should be `CONDITIONING`.
        - `refiner_clip`: unknown Type should be `CLIP`.
        - `latent`: unknown Type should be `LATENT`.
        - `seed`: unknown Type should be `INT`.
        - `width`: unknown Type should be `INT`.
        - `height`: unknown Type should be `INT`.
        - `pos_string`: unknown Type should be `STRING`.
        - `neg_string`: unknown Type should be `STRING`.
