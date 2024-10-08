- `BatchStringSchedule`: The BatchStringSchedule node processes animation prompts by splitting them into positive and negative prompts, interpolating these prompts over a series of frames, and returning the current and next prompts for both positive and negative categories. This node is designed to work with batch processing of strings for animation purposes, facilitating dynamic text generation based on frame-specific settings.
    - Parameters:
        - `text`: The 'text' parameter represents the animation prompts to be processed, serving as the input for splitting and interpolation. Type should be `STRING`.
        - `max_frames`: Specifies the maximum number of frames for which the animation prompts will be processed and interpolated. Type should be `INT`.
        - `print_output`: A flag indicating whether the output should be printed for debugging or logging purposes. Type should be `BOOLEAN`.
        - `pre_text`: Text to be prepended to each animation prompt before processing. Type should be `STRING`.
        - `app_text`: Text to be appended to each animation prompt after processing. Type should be `STRING`.
        - `pw_a`: Parameter weight A for adjusting the interpolation of prompts. Type should be `FLOAT`.
        - `pw_b`: Parameter weight B for further customization of prompt interpolation. Type should be `FLOAT`.
        - `pw_c`: Parameter weight C, used in conjunction with A and B for fine-tuning the interpolation process. Type should be `FLOAT`.
        - `pw_d`: Parameter weight D, provides additional control over the interpolation of animation prompts. Type should be `FLOAT`.
    - Inputs:
    - Outputs:
        - `POS`: The current positive prompt interpolated for the current frame. Type should be `STRING`.
        - `NEG`: The current negative prompt interpolated for the current frame. Type should be `STRING`.
