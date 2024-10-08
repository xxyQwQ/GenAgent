- `Framed Mask Grab Cut 2`: This node applies the GrabCut algorithm with a framing option to segment the foreground from the background in an image. It allows for the specification of areas in the image that are likely to be foreground or background, and can adjust the segmentation based on frame options to exclude certain margins.
    - Parameters:
        - `iterations`: The number of iterations the GrabCut algorithm will run, affecting the accuracy and detail of the segmentation. Type should be `INT`.
        - `margin`: The margin size to exclude from the frame, which can be used to ignore certain edges of the image. Type should be `INT`.
        - `frame_option`: Specifies which margins of the image to exclude from consideration as foreground or background, allowing for more control over the segmentation. Type should be `COMBO[STRING]`.
        - `binary_threshold`: The threshold value used to distinguish between probable foreground and background in the 'thresh_maybe' and 'thresh_sure' images. Type should be `INT`.
        - `maybe_black_is_sure_background`: A flag indicating whether areas identified as probable background in 'thresh_maybe' should be considered definite background, affecting the final segmentation. Type should be `BOOLEAN`.
        - `output_format`: The desired format of the output, which could be a mask indicating the segmented foreground or other formats as needed. Type should be `COMBO[STRING]`.
    - Inputs:
        - `image`: The input image on which the GrabCut algorithm will be applied. This image is preprocessed to fit the algorithm's requirements. Type should be `IMAGE`.
        - `thresh_maybe`: A thresholded version of the image indicating areas that might be the foreground, used to refine the segmentation. Type should be `IMAGE`.
        - `thresh_sure`: A thresholded version of the image indicating areas that are definitely the foreground, further refining the segmentation. Type should be `IMAGE`.
    - Outputs:
        - `image`: The output of the node is an image that has been segmented by the GrabCut algorithm, with the foreground isolated from the background. Type should be `IMAGE`.
