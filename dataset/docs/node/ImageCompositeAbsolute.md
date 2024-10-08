- `ImageCompositeAbsolute`: This node is designed for creating composite images by absolutely positioning two input images within a specified container. It handles the precise placement and blending of these images based on given coordinates, dimensions of the container, and a specified compositing method, facilitating the creation of complex visual layouts.
    - Parameters:
        - `images_a_x`: The x-coordinate for the top-left corner of the first image within the container, defining its horizontal placement. Type should be `INT`.
        - `images_a_y`: The y-coordinate for the top-left corner of the first image within the container, defining its vertical placement. Type should be `INT`.
        - `images_b_x`: The x-coordinate for the top-left corner of the second image within the container, defining its horizontal placement. Type should be `INT`.
        - `images_b_y`: The y-coordinate for the top-left corner of the second image within the container, defining its vertical placement. Type should be `INT`.
        - `container_width`: Specifies the width of the container within which the input images are to be composited. It determines the horizontal boundary for the composition. Type should be `INT`.
        - `container_height`: Specifies the height of the container within which the input images are to be composited. It determines the vertical boundary for the composition. Type should be `INT`.
        - `background`: Determines which of the input images (images_a or images_b) is used as the background in the composite. This choice affects the layering and visual outcome of the final image. Type should be `COMBO[STRING]`.
        - `method`: Specifies the compositing method to be used, influencing how the images are blended together within the container. Type should be `COMBO[STRING]`.
    - Inputs:
        - `images_a`: The first image to be composited within the container. It plays a crucial role in the layering order and visual outcome of the composite image. Type should be `IMAGE`.
        - `images_b`: The second image to be composited within the container, contributing to the complexity and depth of the final composite image. Type should be `IMAGE`.
    - Outputs:
        - `image`: The resulting image after compositing the input images according to the specified parameters and method. Type should be `IMAGE`.
