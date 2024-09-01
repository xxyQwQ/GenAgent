- `ImpactSimpleDetectorSEGS_for_AD`: This node is designed to perform simplified detection tasks specifically tailored for animations and differences in images, leveraging segmentation models to identify and process distinct visual elements.
    - Parameters:
        - `bbox_threshold`: A threshold value to determine the sensitivity of bounding box detection, influencing the identification of distinct elements. Type should be `FLOAT`.
        - `bbox_dilation`: Adjusts the dilation of detected bounding boxes, allowing for more precise control over the segmentation boundaries. Type should be `INT`.
        - `crop_factor`: Determines the factor by which the detected bounding boxes are cropped, affecting the focus area around detected elements. Type should be `FLOAT`.
        - `drop_size`: Specifies the minimum size for detected elements to be considered, filtering out smaller, potentially irrelevant detections. Type should be `INT`.
        - `sub_threshold`: A secondary threshold value for finer control over the detection process, possibly used in conjunction with additional segmentation models. Type should be `FLOAT`.
        - `sub_dilation`: Adjusts the dilation for a secondary detection process, offering further refinement of detected elements' boundaries. Type should be `INT`.
        - `sub_bbox_expansion`: Defines how much to expand the bounding boxes in the secondary detection process, allowing for inclusion of surrounding context. Type should be `INT`.
        - `sam_mask_hint_threshold`: A threshold value for generating mask hints in the SAM model, influencing the model's focus on certain areas of the image. Type should be `FLOAT`.
        - `masking_mode`: Specifies the mode of combining detected segments or masks, affecting how the final segmentation is constructed. Type should be `COMBO[STRING]`.
        - `segs_pivot`: Determines the pivot for segmentation, guiding the combination or selection of segments in the final output. Type should be `COMBO[STRING]`.
    - Inputs:
        - `bbox_detector`: Specifies the bounding box detector to be used for detection, crucial for identifying distinct elements within the animation or image. Type should be `BBOX_DETECTOR`.
        - `image_frames`: The input image frames from an animation to be processed, serving as the primary data for detection tasks. Type should be `IMAGE`.
        - `sam_model_opt`: Optional. Specifies the SAM model to be used for additional mask hint generation, enhancing the detection process. Type should be `SAM_MODEL`.
        - `segm_detector_opt`: Optional. Specifies an additional segmentation detector to be used for refining the detection results. Type should be `SEGM_DETECTOR`.
    - Outputs:
        - `segs`: The output provides segmented elements identified from the input image frames, ready for further processing or analysis. Type should be `SEGS`.