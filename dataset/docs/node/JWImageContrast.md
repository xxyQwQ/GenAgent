- `JWImageContrast`: This node adjusts the contrast of an image based on a specified factor, enhancing or reducing the difference between the light and dark areas of the image.
    - Parameters:
        - `factor`: A multiplier for adjusting the contrast. A factor greater than 1 increases contrast, while a factor less than 1 decreases it. Type should be `FLOAT`.
    - Inputs:
        - `image`: The input image to adjust the contrast for. The adjustment is made by altering the intensity of the pixels. Type should be `IMAGE`.
    - Outputs:
        - `image`: The output image with adjusted contrast. Type should be `IMAGE`.
