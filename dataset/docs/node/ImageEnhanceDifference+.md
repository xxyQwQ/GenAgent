- `ImageEnhanceDifference+`: This node is designed to enhance the difference between two images by applying a power transformation. It aims to highlight the disparities between the images, potentially for further analysis or visual effect enhancement.
    - Parameters:
        - `exponent`: A factor that controls the intensity of the enhancement. Higher values increase the contrast between the differences. Type should be `FLOAT`.
    - Inputs:
        - `image1`: The first image to compare. It serves as the baseline for the enhancement process. Type should be `IMAGE`.
        - `image2`: The second image to compare. This image is adjusted to match the first image's dimensions if necessary, before the difference enhancement. Type should be `IMAGE`.
    - Outputs:
        - `image`: The enhanced difference image, highlighting disparities between the input images. Type should be `IMAGE`.
