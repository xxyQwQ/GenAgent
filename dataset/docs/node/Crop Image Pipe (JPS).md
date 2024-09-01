- `Crop Image Pipe (JPS)`: The Crop Image Pipe node is designed to process image cropping settings, applying specified cropping positions, offsets, and interpolation methods to images. It abstracts the complexity of image cropping operations, enabling users to define how images should be cropped and resized for further processing or visualization.
    - Parameters:
    - Inputs:
        - `cropimage_settings`: Specifies the settings for cropping an image, including the positions, offsets, and interpolation method to be used. This parameter is crucial for determining how the image will be cropped and resized, affecting the final output. Type should be `BASIC_PIPE`.
    - Outputs:
        - `source_crop_pos`: The position from which the source image should be cropped. Type should be `COMBO[STRING]`.
        - `source_crop_offset`: The offset applied to the source image cropping position. Type should be `INT`.
        - `support_crop_pos`: The position from which the support image should be cropped. Type should be `COMBO[STRING]`.
        - `support_crop_offset`: The offset applied to the support image cropping position. Type should be `INT`.
        - `crop_intpol`: The interpolation method used for cropping the image. Type should be `COMBO[STRING]`.