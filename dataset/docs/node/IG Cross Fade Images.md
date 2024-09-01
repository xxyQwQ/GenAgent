- `IG Cross Fade Images`: The IG Cross Fade Images node is designed to create a series of images that smoothly transition from one set to another using cross-fading effects. It leverages easing functions to adjust the transition's pace, allowing for a variety of dynamic visual effects.
    - Parameters:
        - `interpolation`: Determines the easing function used to calculate the alpha values for the cross-fade effect, affecting the transition's dynamics. Type should be `COMBO[STRING]`.
        - `transitioning_frames`: Specifies the number of frames dedicated to transitioning between each pair of images, influencing the smoothness of the cross-fade effect. Type should be `INT`.
        - `repeat_count`: Controls how many times the current image is repeated before transitioning, allowing for customization of the animation's pacing. Type should be `INT`.
    - Inputs:
        - `input_images`: A list of image tensors to be cross-faded. It serves as the primary input for generating the transition effects between images. Type should be `IMAGE`.
    - Outputs:
        - `image`: A tensor containing the sequence of cross-faded images, representing the smooth transition between the input images. Type should be `IMAGE`.