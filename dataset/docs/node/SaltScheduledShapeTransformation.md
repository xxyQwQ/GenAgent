- `SaltScheduledShapeTransformation`: This node is designed for performing scheduled shape transformations on images. It allows for the dynamic adjustment of image shapes over a sequence of frames, based on predefined schedules for various parameters such as size, position, and rotation.
    - Parameters:
        - `max_frames`: Specifies the maximum number of frames for the shape transformation sequence. Type should be `INT`.
        - `image_width`: The width of the output image. Type should be `INT`.
        - `image_height`: The height of the output image. Type should be `INT`.
        - `initial_width`: The initial width of the shape before transformation begins. Type should be `INT`.
        - `initial_height`: The initial height of the shape before transformation begins. Type should be `INT`.
        - `initial_x_coord`: The initial x-coordinate of the shape's position. Type should be `INT`.
        - `initial_y_coord`: The initial y-coordinate of the shape's position. Type should be `INT`.
        - `initial_rotation`: The initial rotation angle of the shape, in degrees. Type should be `FLOAT`.
        - `shape_mode`: Defines the mode or type of shape to be transformed, allowing for various geometric shapes. Type should be `COMBO[STRING]`.
    - Inputs:
        - `shape`: Optional. The specific shape to be transformed, if applicable. Type should be `MASK`.
        - `width_schedule`: A schedule list defining the width transformation over time. Type should be `LIST`.
        - `height_schedule`: A schedule list defining the height transformation over time. Type should be `LIST`.
        - `x_schedule`: A schedule list defining the x-coordinate transformation over time. Type should be `LIST`.
        - `y_schedule`: A schedule list defining the y-coordinate transformation over time. Type should be `LIST`.
        - `rotation_schedule`: A schedule list defining the rotation angle transformation over time. Type should be `LIST`.
    - Outputs:
        - `images`: The transformed images as a result of the scheduled shape transformations. Type should be `IMAGE`.
