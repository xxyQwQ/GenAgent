- `MaskGrid N KSamplers Advanced`: This node specializes in applying advanced sampling techniques to generate or modify latent spaces in a grid format. It leverages multiple samplers and masking strategies to create or alter latents, enabling complex manipulations like forking and merging regions within the grid. The node's functionality is crucial for tasks that require precise control over the spatial distribution of features within generated images or patterns.
    - Parameters:
        - `add_noise`: Indicates whether noise should be added to the sampling process. This parameter can significantly affect the diversity and quality of the generated latents. Type should be `COMBO[STRING]`.
        - `noise_seed`: Sets the seed for noise generation, ensuring reproducibility when adding noise to the sampling process. Type should be `INT`.
        - `steps`: Defines the number of steps to perform during the sampling process. This parameter controls the depth of the sampling operation, affecting the detail and quality of the generated latents. Type should be `INT`.
        - `cfg`: Specifies the conditioning factor for the generative model, influencing the strength of the conditioning during the sampling process. Type should be `FLOAT`.
        - `sampler_name`: Determines the specific sampler algorithm to be used. Different samplers can produce varied effects on the generated latents. Type should be `COMBO[STRING]`.
        - `scheduler`: Selects the scheduler for the sampling process, which can affect the progression and outcome of the sampling operation. Type should be `COMBO[STRING]`.
        - `start_at_step`: Defines the starting step for the sampling process, allowing for more control over the sampling progression. Type should be `INT`.
        - `end_at_step`: Defines the ending step for the sampling process, providing a boundary for the sampling operation. Type should be `INT`.
        - `return_with_leftover_noise`: Indicates whether the output should include leftover noise from the sampling process. This can be useful for further manipulations or analysis. Type should be `COMBO[STRING]`.
        - `rows`: Determines the number of rows in the grid format. This parameter affects the spatial organization of the latents within the grid. Type should be `INT`.
        - `columns`: Determines the number of columns in the grid format. This parameter affects the spatial organization of the latents within the grid. Type should be `INT`.
        - `mode`: Specifies the mode of operation, such as forking before or after sampling. This choice can significantly impact the outcome of the manipulations. Type should be `COMBO[STRING]`.
    - Inputs:
        - `model`: Specifies the model used for sampling. It's essential for defining the generative model's architecture that will be utilized during the sampling process. Type should be `MODEL`.
        - `positive`: Provides positive conditioning inputs to guide the sampling towards desired attributes or features. Type should be `CONDITIONING`.
        - `negative`: Provides negative conditioning inputs to steer the sampling away from certain attributes or features. Type should be `CONDITIONING`.
        - `latent_image`: Represents the initial latent images to be modified through the sampling and masking operations. This parameter is the starting point for the node's complex manipulations. Type should be `LATENT`.
        - `mask`: Specifies the mask to be applied to the latent images, enabling targeted modifications within the grid. Type should be `IMAGE`.
    - Outputs:
        - `latent`: The modified latent images after applying advanced sampling and masking techniques. This output is significant for downstream tasks that rely on the spatially manipulated features within the latents. Type should be `LATENT`.