- `MaskOrImageToWeight`: This node is designed to calculate the mean value of either masks or images provided as input, but not both simultaneously. It supports converting the calculated mean values into different output types, including a list, pandas series, or a tensor, based on the specified output type.
    - Parameters:
        - `output_type`: Specifies the format of the output, which can be a list, pandas series, or tensor, dictating how the mean values calculated from the input masks or images are returned. Type should be `COMBO[STRING]`.
    - Inputs:
        - `images`: An optional list of images to calculate mean values from. If provided, masks should not be used. Type should be `IMAGE`.
        - `masks`: An optional list of masks to calculate mean values from. If provided, images should not be used. Type should be `MASK`.
    - Outputs:
        - `float`: The calculated mean values of the input masks or images, returned in the format specified by the output_type parameter. Type should be `FLOAT`.
