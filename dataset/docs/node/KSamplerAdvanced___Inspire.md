- `KSamplerAdvanced __Inspire`: This node is designed to enhance the sampling process in generative models, specifically tailored for the Inspire pack. It builds upon the standard sampling techniques by incorporating advanced features and optimizations to improve the quality and efficiency of generated samples.
    - Parameters:
        - `add_noise`: Determines whether noise should be added to the sampling process, enhancing the diversity and quality of the generated samples. Type should be `BOOLEAN`.
        - `noise_seed`: Sets the seed for noise generation, ensuring reproducibility and consistency in the samples produced. Type should be `INT`.
        - `steps`: Defines the number of steps to be taken in the sampling process, affecting the detail and quality of the output. Type should be `INT`.
        - `cfg`: Specifies the conditioning-free guidance scale, influencing the direction and strength of the sampling process. Type should be `FLOAT`.
        - `sampler_name`: Identifies the specific sampler algorithm to be used, allowing for customization of the sampling technique. Type should be `COMBO[STRING]`.
        - `scheduler`: Selects the scheduler for controlling the sampling process, further customizing the generation. Type should be `COMBO[STRING]`.
        - `start_at_step`: Specifies the starting step for the sampling process, allowing for mid-process interventions or adjustments. Type should be `INT`.
        - `end_at_step`: Defines the ending step for the sampling process, setting the bounds for generation. Type should be `INT`.
        - `noise_mode`: Determines the computational platform (GPU or CPU) for noise generation, affecting performance and efficiency. Type should be `COMBO[STRING]`.
        - `return_with_leftover_noise`: Indicates whether to return the sample with leftover noise, offering additional control over the output's final appearance. Type should be `BOOLEAN`.
        - `batch_seed_mode`: Specifies the mode for seed generation across batches, affecting the diversity of generated samples. Type should be `COMBO[STRING]`.
        - `variation_seed`: Sets the seed for introducing variations, enabling controlled diversity in the output. Type should be `INT`.
        - `variation_strength`: Determines the strength of variations introduced, allowing for fine-tuning of the output's diversity. Type should be `FLOAT`.
    - Inputs:
        - `model`: Specifies the generative model used for the sampling process, serving as the core component around which the sampling operation revolves. Type should be `MODEL`.
        - `positive`: Provides positive conditioning to guide the sampling towards desired attributes or content. Type should be `CONDITIONING`.
        - `negative`: Provides negative conditioning to steer the sampling away from undesired attributes or content. Type should be `CONDITIONING`.
        - `latent_image`: Inputs a latent image representation to be used or modified during the sampling process. Type should be `LATENT`.
        - `noise_opt`: Provides options for noise customization, offering further control over the sampling process. Type should be `NOISE`.
    - Outputs:
        - `latent`: Outputs a latent representation of the generated sample, encapsulating the result of the advanced sampling process. Type should be `LATENT`.