- `Common Frame Dimensions [Dream]`: The Common Frame Dimensions node provides a utility for calculating frame dimensions based on a set of input parameters including size, aspect ratio, orientation, divisor, and alignment. It abstracts the complexity of dimension calculations and alignment adjustments, offering a streamlined way to determine optimal frame sizes for various display requirements.
    - Parameters:
        - `size`: Specifies the desired frame size from a predefined list of resolutions. This choice influences the overall dimensions of the frame, serving as a base for further calculations. Type should be `COMBO[STRING]`.
        - `aspect_ratio`: Determines the frame's aspect ratio, affecting its width and height proportionally to ensure the specified ratio is maintained. Type should be `COMBO[STRING]`.
        - `orientation`: Indicates the frame's orientation (wide or tall), which influences the calculation of width and height based on the aspect ratio. Type should be `COMBO[STRING]`.
        - `divisor`: A factor used to divide the frame dimensions for finer control over size granularity, affecting the final dimensions. Type should be `COMBO[STRING]`.
        - `alignment`: Specifies the alignment value for dimension calculations, ensuring that the final dimensions are aligned to a certain boundary. Type should be `INT`.
        - `alignment_type`: Determines how the final dimensions are rounded (up, down, or to the nearest) based on the alignment value. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `width`: The calculated width of the frame after considering all input parameters. Type should be `INT`.
        - `height`: The calculated height of the frame after considering all input parameters. Type should be `INT`.
        - `final_width`: The final width of the frame, adjusted according to the alignment and alignment type. Type should be `INT`.
        - `final_height`: The final height of the frame, adjusted according to the alignment and alignment type. Type should be `INT`.
