- `RemapPinch`: The RemapPinch node is designed to apply a pinch distortion effect to images, allowing for the manipulation of image geometry around a specified center point with varying degrees of power.
    - Parameters:
        - `power_x`: Specifies the degree of horizontal pinch effect to apply. Higher values result in more pronounced pinching. Type should be `FLOAT`.
        - `power_y`: Specifies the degree of vertical pinch effect to apply. Higher values result in more pronounced pinching. Type should be `FLOAT`.
        - `center_x`: Determines the horizontal center around which the pinch effect is applied. Values range from 0 to 1, representing the width of the image. Type should be `FLOAT`.
        - `center_y`: Determines the vertical center around which the pinch effect is applied. Values range from 0 to 1, representing the height of the image. Type should be `FLOAT`.
    - Inputs:
    - Outputs:
        - `remap`: This node returns a remap output, which is used to apply the specified pinch distortion effect to the image. Type should be `REMAP`.
