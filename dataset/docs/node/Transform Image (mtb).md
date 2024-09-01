- `Transform Image (mtb)`: The Transform Image (mtb) node is designed to apply a series of affine transformations to an image, including translation, rotation, scaling (zoom), and shearing. It allows for flexible image manipulation with options for border handling and filling with a constant color, making it suitable for a wide range of image processing tasks.
    - Parameters:
        - `x`: The horizontal translation distance. It shifts the image left or right, influencing the image's position. Type should be `FLOAT`.
        - `y`: The vertical translation distance. It shifts the image up or down, affecting the image's vertical positioning. Type should be `FLOAT`.
        - `zoom`: The scaling factor. It enlarges or reduces the image size, impacting the level of detail visible in the transformed image. Type should be `FLOAT`.
        - `angle`: The rotation angle in degrees. It rotates the image around its center, changing its orientation. Type should be `FLOAT`.
        - `shear`: The shear intensity. It distorts the image by slanting it, modifying the shape and angles within the image. Type should be `FLOAT`.
        - `border_handling`: The method used for handling image borders during transformation. It determines how the edges of the image are treated, affecting the appearance of the image's periphery. Type should be `COMBO[STRING]`.
    - Inputs:
        - `image`: The input image tensor to be transformed. It serves as the primary data on which the affine transformations are applied, affecting the visual outcome of the operation. Type should be `IMAGE`.
        - `constant_color`: The color used to fill in new areas when the image is transformed. It specifies the fill color for areas that become exposed as a result of the transformation. Type should be `COLOR`.
    - Outputs:
        - `image`: The transformed image tensor, reflecting the applied affine transformations. Type should be `IMAGE`.