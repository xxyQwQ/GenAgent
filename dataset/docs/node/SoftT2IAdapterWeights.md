- `SoftT2IAdapterWeights`: The SoftT2IAdapterWeights node is designed to adjust the influence of control weights within a text-to-image adaptation process, allowing for a more nuanced and customizable image generation based on the specified weights and the option to flip these weights.
    - Parameters:
        - `weight_i`: Specifies a control weight at index 'i', influencing the adaptation process at various stages. The index 'i' represents a sequence of control weights, allowing for detailed customization of the image generation process. Type should be `FLOAT`.
        - `flip_weights`: A boolean flag that, when true, reverses the order of control weights, potentially altering the adaptation process. Type should be `BOOLEAN`.
    - Inputs:
    - Outputs:
        - `CN_WEIGHTS`: The adjusted control weights after processing through the SoftT2IAdapterWeights node. Type should be `CONTROL_NET_WEIGHTS`.
        - `TK_SHORTCUT`: A keyframe group indicating specific timesteps where the control weights have significant influence. Type should be `TIMESTEP_KEYFRAME`.