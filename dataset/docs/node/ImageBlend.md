- `ImageBlend`: The ImageBlend node is designed to blend two images together based on a specified blending mode and blend factor. It supports various blending modes such as normal, multiply, screen, overlay, soft light, and difference, allowing for versatile image manipulation and compositing techniques. This node is essential for creating composite images by adjusting the visual interaction between two image layers.
    - Parameters:
        - `blend_factor`: Determines the weight of the second image in the blend. A higher blend factor gives more prominence to the second image in the resulting blend. Type should be `FLOAT`.
        - `blend_mode`: Specifies the method of blending the two images. Supports modes like normal, multiply, screen, overlay, soft light, and difference, each producing a unique visual effect. Type should be `COMBO[STRING]`.
    - Inputs:
        - `image1`: The first image to be blended. It serves as the base layer for the blending operation. Type should be `IMAGE`.
        - `image2`: The second image to be blended. Depending on the blend mode, it modifies the appearance of the first image. Type should be `IMAGE`.
    - Outputs:
        - `image`: The resulting image after blending the two input images according to the specified blend mode and factor. Type should be `IMAGE`.
