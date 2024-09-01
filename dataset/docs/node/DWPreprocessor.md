- `DWPreprocessor`: The DWPreprocessor node is designed for preprocessing input data specifically for the DWPose estimation tasks. It transforms input data into a format suitable for pose estimation, enhancing the performance of pose estimation models by optimizing the input data's structure and format.
    - Parameters:
        - `detect_hand`: Enables or disables hand detection in the pose estimation process, affecting the comprehensiveness of the pose analysis. Type should be `COMBO[STRING]`.
        - `detect_body`: Enables or disables body detection, determining whether body keypoints are included in the pose estimation. Type should be `COMBO[STRING]`.
        - `detect_face`: Controls the inclusion of face detection in the pose estimation, influencing the detail level of facial keypoints. Type should be `COMBO[STRING]`.
        - `resolution`: The resolution to which the input image is resized, affecting the detail level of the pose estimation. Type should be `INT`.
        - `bbox_detector`: Specifies the bounding box detector model to use, impacting the initial detection phase of pose estimation. Type should be `COMBO[STRING]`.
        - `pose_estimator`: Determines the pose estimation model, directly affecting the accuracy and performance of pose keypoint detection. Type should be `COMBO[STRING]`.
    - Inputs:
        - `image`: The input image to be processed for pose estimation. Type should be `IMAGE`.
    - Outputs:
        - `image`: The processed image after pose estimation, ready for further analysis or visualization. Type should be `IMAGE`.
        - `pose_keypoint`: The detected pose keypoints, providing detailed positional information for body parts. Type should be `POSE_KEYPOINT`.