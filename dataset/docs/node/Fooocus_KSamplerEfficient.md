- `Fooocus_KSamplerEfficient`: The Fooocus_KSamplerEfficient node enhances the sampling process in art generation by incorporating a sharpness parameter, allowing for more precise control over the texture and detail level of generated images. This node builds upon the foundational sampling capabilities to offer an advanced, efficiency-focused approach to art creation.
    - Parameters:
        - `seed`: The seed parameter ensures reproducibility in the art generation process by initializing the random number generator to a specific state. Type should be `INT`.
        - `steps`: Defines the number of steps in the sampling process, affecting the detail and quality of the generated art. Type should be `INT`.
        - `cfg`: Configures the conditioning factor for the sampling process, influencing the generation's creativity and coherence. Type should be `FLOAT`.
        - `sampler_name`: Identifies the specific sampler algorithm to be used, affecting the texture and detail of the generated art. Type should be `COMBO[STRING]`.
        - `scheduler`: Specifies the scheduler for controlling the sampling process, impacting the progression and quality of art generation. Type should be `COMBO[STRING]`.
        - `denoise`: Adjusts the level of denoising applied to the generated art, affecting clarity and detail. Type should be `FLOAT`.
        - `preview_method`: unknown Type should be `COMBO[STRING]`.
        - `vae_decode`: unknown Type should be `COMBO[STRING]`.
        - `sharpness`: The sharpness parameter allows users to adjust the level of detail and texture in the generated art, providing a means to fine-tune the visual output for more precise artistic control. Type should be `FLOAT`.
    - Inputs:
        - `model`: Specifies the model used for the sampling process, integral to determining the art generation's foundational style and characteristics. Type should be `MODEL`.
        - `positive`: Defines positive conditioning to guide the art generation towards desired attributes. Type should be `CONDITIONING`.
        - `negative`: Sets negative conditioning to avoid certain attributes in the generated art. Type should be `CONDITIONING`.
        - `latent_image`: Provides the initial latent image to be transformed by the sampling process. Type should be `LATENT`.
        - `optional_vae`: unknown Type should be `VAE`.
        - `script`: unknown Type should be `SCRIPT`.
    - Outputs:
        - `MODEL`: unknown Type should be `MODEL`.
        - `CONDITIONING+`: unknown Type should be `CONDITIONING`.
        - `CONDITIONING-`: unknown Type should be `CONDITIONING`.
        - `LATENT`: The output latent image represents the final generated art, encapsulating the visual characteristics specified through the input parameters. Type should be `LATENT`.
        - `VAE`: unknown Type should be `VAE`.
        - `IMAGE`: unknown Type should be `IMAGE`.
