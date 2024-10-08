- `ADE_CameraPoseBasic`: This node is designed to create basic camera control poses based on specified motion types, speed, and frame length. It allows for the generation of camera poses that can be used to control the camera's movement and orientation in a 3D environment, facilitating dynamic and customizable animations.
    - Parameters:
        - `motion_type`: Specifies the type of motion to apply to the camera, influencing the direction and nature of the camera's movement. Type should be `COMBO[STRING]`.
        - `speed`: Determines the speed at which the camera moves, allowing for control over the pace of the animation. Type should be `FLOAT`.
        - `frame_length`: Defines the number of frames over which the camera motion is applied, setting the duration of the camera's movement. Type should be `INT`.
    - Inputs:
        - `prev_poses`: Optional. Provides previous camera poses to be combined with the newly generated ones, enabling seamless transitions between animations. Type should be `CAMERACTRL_POSES`.
    - Outputs:
        - `cameractrl_poses`: Outputs the generated camera control poses, ready for use in animations. Type should be `CAMERACTRL_POSES`.
