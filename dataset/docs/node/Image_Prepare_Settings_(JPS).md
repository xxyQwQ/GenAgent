- `Image Prepare Settings (JPS)`: This node is designed to configure and prepare image settings for further processing or transformation. It allows for the customization of various parameters such as resizing, cropping, padding, and applying specific image adjustments like sharpening or interpolation, facilitating tailored image preparation workflows.
    - Parameters:
        - `offset_width`: The horizontal offset applied to the image, useful for precise positioning or adjustments. Type should be `INT`.
        - `offset_height`: The vertical offset applied to the image, useful for precise positioning or adjustments. Type should be `INT`.
        - `crop_left`: The amount of cropping from the left side of the image, allowing for tailored image composition. Type should be `INT`.
        - `crop_right`: The amount of cropping from the right side of the image, allowing for tailored image composition. Type should be `INT`.
        - `crop_top`: The amount of cropping from the top of the image, allowing for tailored image composition. Type should be `INT`.
        - `crop_bottom`: The amount of cropping from the bottom of the image, allowing for tailored image composition. Type should be `INT`.
        - `padding_left`: The amount of padding added to the left side of the image, useful for framing or specific layout requirements. Type should be `INT`.
        - `padding_right`: The amount of padding added to the right side of the image, useful for framing or specific layout requirements. Type should be `INT`.
        - `padding_top`: The amount of padding added to the top of the image, useful for framing or specific layout requirements. Type should be `INT`.
        - `padding_bottom`: The amount of padding added to the bottom of the image, useful for framing or specific layout requirements. Type should be `INT`.
        - `interpolation`: Specifies the interpolation method used during resizing or transforming the image, affecting the image's smoothness and quality. Type should be `COMBO[STRING]`.
        - `sharpening`: The level of sharpening applied to the image, enhancing detail and clarity. Type should be `FLOAT`.
    - Inputs:
    - Outputs:
        - `imageprepare_settings`: The configured settings for image preparation, encapsulating all adjustments and transformations to be applied. Type should be `BASIC_PIPE`.