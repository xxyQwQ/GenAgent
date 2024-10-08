- `KSamplerProvider`: The KSamplerProvider node is designed to facilitate the creation of custom samplers for generative models, allowing users to specify various parameters such as seed, steps, configuration settings, and the type of sampler and scheduler to be used. It abstracts the complexity of sampler initialization and configuration, making it easier to experiment with different sampling strategies.
    - Parameters:
        - `seed`: The seed parameter ensures reproducibility of the sampling process by initializing the random number generator with a specific value. Type should be `INT`.
        - `steps`: Defines the number of steps the sampler will take, affecting the detail and quality of the generated samples. Type should be `INT`.
        - `cfg`: Configuration setting that influences the behavior of the sampler, potentially affecting aspects like sample diversity. Type should be `FLOAT`.
        - `sampler_name`: Specifies the type of sampler to use, allowing for customization of the sampling process. Type should be `COMBO[STRING]`.
        - `scheduler`: Determines the scheduling algorithm to be used, impacting how sampling parameters are adjusted over time. Type should be `COMBO[STRING]`.
        - `denoise`: Controls the level of denoising applied to the samples, affecting their clarity and sharpness. Type should be `FLOAT`.
    - Inputs:
        - `basic_pipe`: A foundational pipeline component that provides essential model and conditioning information for the sampling process. Type should be `BASIC_PIPE`.
    - Outputs:
        - `ksampler`: Produces a custom sampler configured according to the specified parameters, ready for generating samples. Type should be `KSAMPLER`.
