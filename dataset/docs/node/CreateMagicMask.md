- `CreateMagicMask`: The CreateMagicMask node is designed to generate dynamic masks based on a set of parameters including frames, transitions, depth, distortion, seed, frame width, and frame height. These masks can be used to apply various visual effects to images or video frames, enabling creative and complex image processing tasks.
    - Parameters:
        - `frames`: Specifies the number of frames for which the mask will be generated, affecting the temporal length of the resulting mask sequence. Type should be `INT`.
        - `depth`: Determines the depth of the mask effect, impacting the perceived three-dimensionality or layering within the mask. Type should be `INT`.
        - `distortion`: Controls the level of distortion applied to the mask, allowing for varied degrees of visual warping and alteration. Type should be `FLOAT`.
        - `seed`: Sets the random seed for mask generation, ensuring reproducibility of the mask patterns. Type should be `INT`.
        - `transitions`: Defines the types or patterns of transitions between masks, influencing the visual flow and changes across the generated frames. Type should be `INT`.
        - `frame_width`: Specifies the width of the frames, defining the horizontal dimension of the generated masks. Type should be `INT`.
        - `frame_height`: Specifies the height of the frames, defining the vertical dimension of the generated masks. Type should be `INT`.
    - Inputs:
    - Outputs:
        - `mask`: Generates a dynamic mask based on the specified input parameters. Type should be `MASK`.
        - `mask_inverted`: Generates an inverted version of the dynamic mask, where the mask values are reversed. Type should be `MASK`.
