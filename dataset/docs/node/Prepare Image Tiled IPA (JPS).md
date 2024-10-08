- `Prepare Image Tiled IPA (JPS)`: The Prepare Image Tiled IPA node is designed to configure and apply image preparation settings for tiled image processing. It adjusts parameters such as model type, weight type, noise level, and image tiling options based on the input specifications, facilitating the tailored preprocessing of images for enhanced image synthesis or manipulation tasks.
    - Parameters:
        - `target_w`: Sets the target width for the image processing, influencing the dimensions of the output image. Type should be `INT`.
        - `target_h`: Defines the target height for the image processing, influencing the dimensions of the output image. Type should be `INT`.
        - `zoom`: Determines the zoom level applied to the image, affecting the scale of the image details. Type should be `INT`.
        - `offset_w`: Specifies the horizontal offset for the image, adjusting its position along the width. Type should be `INT`.
        - `offset_h`: Specifies the vertical offset for the image, adjusting its position along the height. Type should be `INT`.
        - `interpolation`: Chooses the interpolation method for image resizing, affecting the smoothness and quality of the resized image. Type should be `COMBO[STRING]`.
        - `sharpening`: Sets the level of sharpening to be applied to the image, enhancing edge definition and detail. Type should be `FLOAT`.
        - `tile_short`: Defines the base length of the shortest side of the tile, influencing the tiling pattern and size. Type should be `INT`.
        - `prepare_type`: Selects the preparation type for the image, determining the specific processing approach to be applied. Type should be `INT`.
    - Inputs:
        - `image`: Specifies the image to be processed, serving as the base for all subsequent image preparation operations. Type should be `IMAGE`.
    - Outputs:
        - `IMAGE`: Returns the processed image with applied tiling, interpolation, and sharpening settings, ready for further use or analysis. Type should be `IMAGE`.
