- `BNK_InjectNoise`: The `InjectNoise` node is designed to augment latent representations with noise, offering a mechanism to introduce variability and potentially enhance the generation quality of models. It supports the injection of custom noise and the application of a mask to selectively apply noise, thereby providing a versatile tool for manipulating latent spaces.
    - Parameters:
        - `strength`: Determines the intensity of the noise applied to the latents. A higher value results in more pronounced noise effects. Type should be `FLOAT`.
    - Inputs:
        - `latents`: The primary latent representation to which noise will be added. This parameter is crucial for defining the base structure that the noise will modify. Type should be `LATENT`.
        - `noise`: An optional parameter allowing for the injection of custom noise into the latent representation. Type should be `LATENT`.
        - `mask`: An optional mask that can be applied to selectively introduce noise to specific areas of the latent representation. Type should be `MASK`.
    - Outputs:
        - `latent`: The modified latent representation after noise injection, potentially with selective application based on an optional mask. Type should be `LATENT`.