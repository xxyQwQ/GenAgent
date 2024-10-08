- `Inference_Core_FacialPartColoringFromPoseKps`: This node is designed to colorize specific facial parts in pose keypoint data, utilizing customizable colors for each facial part. It processes pose keypoint frames to visually enhance and distinguish different facial regions based on the provided keypoint data and color specifications.
    - Parameters:
        - `mode`: Specifies the drawing mode for keypoints, allowing for either point-based or polygon-based rendering of facial parts. Type should be `COMBO[STRING]`.
        - `skin`: Specifies the color for the skin facial part, affecting the visual representation of the skin area in the output. Type should be `STRING`.
        - `left_eye`: Specifies the color for the left eye, affecting how the left eye is visualized in the output. Type should be `STRING`.
        - `right_eye`: Specifies the color for the right eye, affecting how the right eye is visualized in the output. Type should be `STRING`.
        - `nose`: Specifies the color for the nose, affecting how the nose is visualized in the output. Type should be `STRING`.
        - `upper_lip`: Specifies the color for the upper lip, affecting how the upper lip is visualized in the output. Type should be `STRING`.
        - `inner_mouth`: Specifies the color for the inner mouth, affecting how the inner mouth is visualized in the output. Type should be `STRING`.
        - `lower_lip`: Specifies the color for the lower lip, affecting how the lower lip is visualized in the output. Type should be `STRING`.
    - Inputs:
        - `pose_kps`: The pose keypoint data containing information about the positions of various facial parts. It serves as the primary input for generating colorized facial keypoints. Type should be `POSE_KEYPOINT`.
    - Outputs:
        - `image`: The output is a tensor representation of the pose frames with facial parts colorized according to the specified colors. Type should be `IMAGE`.
