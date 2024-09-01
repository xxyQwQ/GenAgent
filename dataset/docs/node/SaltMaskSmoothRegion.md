- `SaltMaskSmoothRegion`: The SaltMaskSmoothRegion node applies a smoothing filter to regions within masks, utilizing a specified sigma value to control the smoothness level. This process enhances the visual quality of mask regions by reducing noise and irregularities.
    - Parameters:
        - `sigma`: The 'sigma' parameter controls the smoothness level of the smoothing filter applied to the mask regions, directly influencing the degree of smoothing and noise reduction. Type should be `FLOAT`.
    - Inputs:
        - `masks`: The 'masks' parameter represents the input masks on which the smoothing operation is to be performed, serving as the primary data for the node's processing. Type should be `MASK`.
    - Outputs:
        - `MASKS`: The output is a tensor of masks that have been smoothed according to the specified sigma value, offering enhanced visual quality by reducing noise and irregularities. Type should be `MASK`.