- `ImageFilterBlur`: The ImageFilterBlur node applies a simple blurring effect to images using a specified horizontal and vertical size. This node is designed to soften images, reducing detail and noise by averaging the pixels within the defined kernel size.
    - Parameters:
        - `size_x`: Specifies the horizontal size of the blur kernel. This size influences the extent of blurring in the horizontal direction. Type should be `INT`.
        - `size_y`: Specifies the vertical size of the blur kernel. This size influences the extent of blurring in the vertical direction. Type should be `INT`.
    - Inputs:
        - `images`: The input images to be blurred. This parameter is crucial for defining the source images on which the blur effect will be applied. Type should be `IMAGE`.
    - Outputs:
        - `image`: The output images after applying the blur effect. This shows the result of the blurring process on the input images. Type should be `IMAGE`.
