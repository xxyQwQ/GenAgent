- `ADE_NoiseLayerAdd`: The `ADE_NoiseLayerAdd` node is designed to integrate an additive noise layer into a given noise structure, enhancing the variability and complexity of the noise pattern. This node facilitates the dynamic adjustment of noise characteristics in generative models, allowing for more nuanced and controlled modifications to the generated outputs.
    - Parameters:
        - `batch_offset`: Specifies the offset for batch processing, affecting how noise is applied across different batches, thereby enabling more precise control over the noise application process. Type should be `INT`.
        - `noise_type`: Defines the type of noise to be added, determining the characteristics and behavior of the noise layer within the generative process. Type should be `COMBO[STRING]`.
        - `seed_gen_override`: Allows for the override of the default seed generation mechanism, enabling custom seed generation strategies for noise creation. Type should be `COMBO[STRING]`.
        - `seed_offset`: Provides an additional offset to the seed value, further customizing the noise generation process for enhanced variability. Type should be `INT`.
        - `noise_weight`: Determines the weight of the added noise, influencing the intensity and impact of the noise on the final output. Type should be `FLOAT`.
        - `seed_override`: Optionally overrides the seed used for noise generation, providing direct control over the randomness of the noise. Type should be `INT`.
    - Inputs:
        - `prev_noise_layers`: Optional parameter to specify previous noise layers, enabling the stacking and combination of multiple noise layers for complex noise structures. Type should be `NOISE_LAYERS`.
        - `mask_optional`: An optional mask to selectively apply noise, allowing for targeted noise application in specific areas of interest. Type should be `MASK`.
    - Outputs:
        - `noise_layers`: Returns the updated noise layer group, including the newly added additive noise layer, facilitating the construction of layered noise structures. Type should be `NOISE_LAYERS`.
