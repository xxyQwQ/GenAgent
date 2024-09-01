- `ttN pipeKSamplerAdvanced_v2`: The `ttN_pipeKSamplerAdvanced_v2` node is designed to enhance the sampling process in generative models by incorporating advanced techniques for noise management and image quality improvement. It leverages a variety of parameters to fine-tune the generation process, aiming to produce higher-quality outputs with greater control over the sampling dynamics.
    - Parameters:
        - `lora_name`: unknown Type should be `COMBO[STRING]`.
        - `lora_strength`: unknown Type should be `FLOAT`.
        - `upscale_method`: unknown Type should be `COMBO[STRING]`.
        - `upscale_model_name`: unknown Type should be `COMBO[STRING]`.
        - `factor`: unknown Type should be `FLOAT`.
        - `rescale`: unknown Type should be `COMBO[STRING]`.
        - `percent`: unknown Type should be `INT`.
        - `width`: unknown Type should be `INT`.
        - `height`: unknown Type should be `INT`.
        - `longer_side`: unknown Type should be `INT`.
        - `crop`: unknown Type should be `COMBO[STRING]`.
        - `sampler_state`: unknown Type should be `COMBO[STRING]`.
        - `add_noise`: unknown Type should be `COMBO[STRING]`.
        - `noise`: unknown Type should be `FLOAT`.
        - `steps`: unknown Type should be `INT`.
        - `start_at_step`: unknown Type should be `INT`.
        - `end_at_step`: unknown Type should be `INT`.
        - `cfg`: unknown Type should be `FLOAT`.
        - `sampler_name`: unknown Type should be `COMBO[STRING]`.
        - `scheduler`: unknown Type should be `COMBO[STRING]`.
        - `return_with_leftover_noise`: unknown Type should be `COMBO[STRING]`.
        - `image_output`: unknown Type should be `COMBO[STRING]`.
        - `save_prefix`: unknown Type should be `STRING`.
        - `noise_seed`: unknown Type should be `INT`.
    - Inputs:
        - `pipe`: unknown Type should be `PIPE_LINE`.
        - `optional_model`: unknown Type should be `MODEL`.
        - `optional_positive`: unknown Type should be `CONDITIONING`.
        - `optional_negative`: unknown Type should be `CONDITIONING`.
        - `optional_latent`: unknown Type should be `LATENT`.
        - `optional_vae`: unknown Type should be `VAE`.
        - `optional_clip`: unknown Type should be `CLIP`.
        - `input_image_override`: unknown Type should be `IMAGE`.
        - `adv_xyPlot`: unknown Type should be `ADV_XYPLOT`.
    - Outputs:
        - `pipe`: unknown Type should be `PIPE_LINE`.
        - `model`: unknown Type should be `MODEL`.
        - `positive`: unknown Type should be `CONDITIONING`.
        - `negative`: unknown Type should be `CONDITIONING`.
        - `latent`: unknown Type should be `LATENT`.
        - `vae`: unknown Type should be `VAE`.
        - `clip`: unknown Type should be `CLIP`.
        - `image`: unknown Type should be `IMAGE`.
        - `seed`: unknown Type should be `INT`.