- `ControlNetHadamard (manual)`: This node applies a control network to a set of images based on provided conditions and a specified strength, adjusting the images' conditioning for further processing. It leverages the concept of Hadamard product in the context of control networks to modulate the influence of the control network on the images.
    - Parameters:
        - `strength`: A scalar value determining the intensity of the control network's effect on the images. Type should be `FLOAT`.
        - `inputs_len`: Specifies the number of images to be processed, allowing for dynamic adjustment based on input. Type should be `INT`.
    - Inputs:
        - `conds`: Conditions to apply to each image, determining how the control network modifies the image. Type should be `CONDITIONING`.
        - `control_net`: The control network to be applied to the images, dictating the nature of the modifications. Type should be `CONTROL_NET`.
    - Outputs:
        - `conditioning`: The modified conditions after applying the control network to the images. Type should be `CONDITIONING`.