- `PoseNode`: The PoseNode is designed to process images to generate their pose representations. It leverages image processing techniques to convert images into a format suitable for pose analysis, abstracting the complexity of image manipulation and conversion for pose detection tasks.
    - Parameters:
        - `image`: The 'image' parameter specifies the image file to be processed for pose detection. It plays a crucial role in the node's operation by serving as the primary input from which pose information is derived. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `image`: The output is a tensor representation of the processed image, suitable for further analysis or visualization in pose detection tasks. Type should be `IMAGE`.