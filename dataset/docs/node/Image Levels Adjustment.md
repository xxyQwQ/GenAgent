- `Image Levels Adjustment`: The Image Levels Adjustment node is designed to modify the tonal range of an image by adjusting its black, mid, and white levels. This process enhances the visual quality of the image by altering its contrast and brightness, making it more visually appealing or suitable for further processing.
    - Parameters:
        - `black_level`: Specifies the minimum intensity value that pixels in the image should have. Adjusting this level affects the overall darkness of the image, enhancing details in darker regions. Type should be `FLOAT`.
        - `mid_level`: Defines the midpoint intensity value for the image's tonal range. Adjusting the mid level can significantly alter the image's contrast, affecting its overall visual appearance. Type should be `FLOAT`.
        - `white_level`: Sets the maximum intensity value that pixels in the image can reach. This level adjustment brightens the image and can bring out details in lighter areas. Type should be `FLOAT`.
    - Inputs:
        - `image`: The input image to be adjusted. This parameter is crucial as it serves as the base for the levels adjustment process, directly influencing the outcome of the adjustment. Type should be `IMAGE`.
    - Outputs:
        - `image`: The adjusted image with modified black, mid, and white levels. This output reflects the changes made to the image's tonal range, enhancing its visual quality. Type should be `IMAGE`.
