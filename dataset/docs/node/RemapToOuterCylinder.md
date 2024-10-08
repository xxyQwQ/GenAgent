- `RemapToOuterCylinder`: This node is designed to transform an image by mapping it onto the surface of an imaginary outer cylinder. It adjusts the image's perspective to simulate how it would appear wrapped around a cylinder, taking into account the field of view and whether to swap the x and y coordinates.
    - Parameters:
        - `fov`: Specifies the field of view in degrees, influencing the curvature and perspective of the remapped image. Type should be `INT`.
        - `swap_xy`: Determines whether the x and y coordinates should be swapped, affecting the orientation of the remapped image. Type should be `BOOLEAN`.
    - Inputs:
    - Outputs:
        - `remap`: Provides the mappings for x and y coordinates to the original image pixels, facilitating the transformation of the image onto the outer cylinder's surface. Type should be `REMAP`.
