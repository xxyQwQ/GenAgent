- `Noise Control Script`: The TSC_Noise_Control_Script node is designed to integrate and manage noise sources and seed variations within the ComfyUI framework. It allows for the customization of noise generation and seed manipulation, enhancing the control over the randomness and variability in generated outputs. This node serves as a pivotal element in adjusting the noise characteristics and seed behavior, thereby offering a nuanced approach to managing the stochastic elements of the generation process.
    - Parameters:
        - `rng_source`: Specifies the source of randomness for noise generation, allowing selection from CPU, GPU, or NV options. This choice influences the computational backend for noise generation, impacting performance and compatibility. Type should be `COMBO[STRING]`.
        - `cfg_denoiser`: A boolean flag that enables or disables the configuration denoiser, affecting the noise filtering process during generation. Type should be `BOOLEAN`.
        - `add_seed_noise`: Determines whether additional noise based on a seed value should be introduced, adding another layer of variability to the output. Type should be `BOOLEAN`.
        - `seed`: The seed value for noise generation, providing a basis for reproducibility and variation in the noise applied. Type should be `INT`.
        - `weight`: Defines the weight of the seed noise, adjusting the intensity of the noise effect on the output. Type should be `FLOAT`.
    - Inputs:
        - `script`: An optional script parameter that can be modified or extended with noise settings, offering flexibility in script-based customization. Type should be `SCRIPT`.
    - Outputs:
        - `SCRIPT`: Returns a modified or extended script with applied noise settings, facilitating customized noise control within the generation process. Type should be `SCRIPT`.