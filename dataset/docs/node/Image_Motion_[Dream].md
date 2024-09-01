- `Image Motion [Dream]`: The Image Motion node applies a series of transformations to an image to simulate motion, including zooming and translating in both x and y directions, along with applying overlap masks for more complex motion effects.
    - Parameters:
        - `zoom`: Defines the zoom level for the image, allowing for both zoom-in and zoom-out effects to simulate motion. Type should be `FLOAT`.
        - `mask_i_feather`: unknown Type should be `INT`.
        - `mask_i_overlap`: unknown Type should be `INT`.
        - `x_translation`: Controls the horizontal movement of the image, simulating lateral motion. Type should be `FLOAT`.
        - `y_translation`: Manages the vertical movement of the image, simulating upward or downward motion. Type should be `FLOAT`.
        - `output_resize_width`: Optionally resizes the output image's width, affecting the final image dimensions. Type should be `INT`.
        - `output_resize_height`: Optionally resizes the output image's height, affecting the final image dimensions. Type should be `INT`.
    - Inputs:
        - `image`: The primary image to which motion effects will be applied. It serves as the base for all subsequent transformations. Type should be `IMAGE`.
        - `frame_counter`: Tracks the number of frames processed, used for animations or effects that change over time. Type should be `FRAME_COUNTER`.
        - `noise`: An optional image used to add noise to the motion effect, enhancing realism. Type should be `IMAGE`.
    - Outputs:
        - `image`: The transformed image with applied motion effects. Type should be `IMAGE`.
        - `mask1`: The first mask used in creating the motion effect, after transformations. Type should be `MASK`.
        - `mask2`: The second mask applied for enhanced motion effects, post-transformation. Type should be `MASK`.
        - `mask3`: The third mask contributing to the layered motion effect, following transformations. Type should be `MASK`.