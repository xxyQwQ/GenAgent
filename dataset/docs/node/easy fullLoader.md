- `easy fullLoader`: The `easy fullLoader` node is designed to streamline the process of loading and configuring various components necessary for running Stable Diffusion models. It abstracts the complexities involved in setting up models, VAEs, CLIP models, and other configurations, providing a simplified interface for users to quickly get started with generating images.
    - Parameters:
        - `ckpt_name`: Specifies the checkpoint name for the model to be loaded, playing a crucial role in determining the model's weights and behavior. Type should be `COMBO[STRING]`.
        - `config_name`: unknown Type should be `COMBO[STRING]`.
        - `vae_name`: Indicates the name of the VAE (Variational Autoencoder) to be used, which is essential for the image generation process. Type should be `COMBO[STRING]`.
        - `clip_skip`: A flag to skip loading the CLIP model, which can be useful for optimizing performance in certain scenarios. Type should be `INT`.
        - `lora_name`: Specifies the name of the LoRA (Low-Rank Adaptation) model to be used, if any, for enhancing model performance. Type should be `COMBO[STRING]`.
        - `lora_model_strength`: Determines the strength of the LoRA model's influence on the overall model performance. Type should be `FLOAT`.
        - `lora_clip_strength`: Sets the strength of the LoRA model's influence specifically on the CLIP model's performance. Type should be `FLOAT`.
        - `resolution`: Defines the resolution of the generated images, directly affecting the output quality. Type should be `COMBO[STRING]`.
        - `empty_latent_width`: Specifies the width of the latent space to be used for image generation, impacting the aspect ratio of the output. Type should be `INT`.
        - `empty_latent_height`: Specifies the height of the latent space, affecting the vertical dimension of the generated images. Type should be `INT`.
        - `positive`: A positive prompt to guide the image generation towards desired attributes or themes. Type should be `STRING`.
        - `positive_token_normalization`: unknown Type should be `COMBO[STRING]`.
        - `positive_weight_interpretation`: unknown Type should be `COMBO[STRING]`.
        - `negative`: A negative prompt to steer the image generation away from certain attributes or themes. Type should be `STRING`.
        - `negative_token_normalization`: unknown Type should be `COMBO[STRING]`.
        - `negative_weight_interpretation`: unknown Type should be `COMBO[STRING]`.
        - `batch_size`: Determines the number of images to be generated in a single batch, influencing performance and output volume. Type should be `INT`.
        - `a1111_prompt_style`: Enables the use of a specific prompt style, potentially altering the image generation process. Type should be `BOOLEAN`.
    - Inputs:
        - `model_override`: unknown Type should be `MODEL`.
        - `clip_override`: unknown Type should be `CLIP`.
        - `vae_override`: unknown Type should be `VAE`.
        - `optional_lora_stack`: Allows for the specification of an optional LoRA stack to further customize model performance. Type should be `LORA_STACK`.
    - Outputs:
        - `pipe`: Returns the configured pipeline for image generation. Type should be `PIPE_LINE`.
        - `model`: Provides the loaded model configured for image generation. Type should be `MODEL`.
        - `vae`: Returns the loaded VAE used in the image generation process. Type should be `VAE`.
        - `clip`: Provides the loaded CLIP model, if not skipped, used for text-to-image encoding. Type should be `CLIP`.
        - `positive`: Returns the positive embeddings generated from the positive prompt. Type should be `CONDITIONING`.
        - `negative`: Returns the negative embeddings generated from the negative prompt. Type should be `CONDITIONING`.
        - `latent`: Provides the latent space configuration used for image generation. Type should be `LATENT`.
