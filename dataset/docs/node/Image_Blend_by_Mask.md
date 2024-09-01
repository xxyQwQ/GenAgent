- `Image Blend by Mask`: The Image Blend by Mask node is designed to blend two images together based on a mask and a specified blend percentage. It allows for the creation of composite images by selectively blending parts of two input images according to the mask's pattern, offering a high degree of control over the blending process.
    - Parameters:
        - `blend_percentage`: A float value that specifies the percentage of blending between the two images. A higher value increases the visibility of the second image in the blended result. Type should be `FLOAT`.
    - Inputs:
        - `image_a`: The first image to be blended. It serves as the base layer in the blending operation. Type should be `IMAGE`.
        - `image_b`: The second image to be blended. It is combined with the first image according to the mask and blend percentage. Type should be `IMAGE`.
        - `mask`: A mask image that determines the blending pattern. Areas in white allow the second image to show through, while black areas retain the first image. Type should be `IMAGE`.
    - Outputs:
        - `image`: The resulting image after blending the two input images based on the mask and blend percentage. Type should be `IMAGE`.