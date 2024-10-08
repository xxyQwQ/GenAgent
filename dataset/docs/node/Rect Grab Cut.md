- `Rect Grab Cut`: The RectGrabCut node is designed for image segmentation using the GrabCut algorithm with a predefined rectangular area. It aims to separate the foreground from the background within the specified rectangle, enhancing the focus on the desired object or area in an image.
    - Parameters:
        - `x1`: The x-coordinate of the top left corner of the rectangle. Type should be `INT`.
        - `y1`: The y-coordinate of the top left corner of the rectangle. Type should be `INT`.
        - `x2`: The x-coordinate of the bottom right corner of the rectangle. Type should be `INT`.
        - `y2`: The y-coordinate of the bottom right corner of the rectangle. Type should be `INT`.
        - `iterations`: The number of iterations the GrabCut algorithm should run to refine the segmentation. Type should be `INT`.
        - `output_format`: The format in which the segmented image should be outputted, affecting how the image is processed and displayed post-segmentation. Type should be `COMBO[STRING]`.
    - Inputs:
        - `image`: The input image to be segmented. This image is processed to separate the foreground from the background within a specified rectangular area. Type should be `IMAGE`.
    - Outputs:
        - `image`: The segmented image with the foreground separated from the background within the specified rectangular area. Type should be `IMAGE`.
