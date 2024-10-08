- `StringSchedule`: The StringSchedule node is designed to manage and schedule string-based content for dynamic animation or content generation. It leverages scheduling settings to interpolate between different string values over a series of frames, facilitating the creation of animated text or evolving narratives.
    - Parameters:
        - `text`: This input takes a multiline string that represents the text to be scheduled and animated, serving as the primary content for the scheduling process. Type should be `STRING`.
        - `max_frames`: Specifies the maximum number of frames for the scheduling, determining the length of the animation or content generation process. Type should be `INT`.
        - `current_frame`: Indicates the current frame number in the scheduling process, used to calculate the specific string content to be displayed at any given frame. Type should be `INT`.
        - `print_output`: A boolean flag that, when set to true, enables the printing of the scheduling process's output for debugging or tracking purposes. Type should be `BOOLEAN`.
        - `pre_text`: An optional string input that is prepended to the main text before scheduling. Type should be `STRING`.
        - `app_text`: An optional string input that is appended to the main text after scheduling. Type should be `STRING`.
        - `pw_a`: A parameter weight used in the scheduling algorithm to adjust the influence of certain aspects of the text. Type should be `FLOAT`.
        - `pw_b`: A parameter weight similar to pw_a, used for adjusting the scheduling algorithm's influence on the text. Type should be `FLOAT`.
        - `pw_c`: Another parameter weight for fine-tuning the scheduling algorithm's effect on the text. Type should be `FLOAT`.
        - `pw_d`: A parameter weight that works alongside pw_a, pw_b, and pw_c to customize the scheduling outcome. Type should be `FLOAT`.
    - Inputs:
    - Outputs:
        - `POS`: The output is a dynamically scheduled string, representing the positive aspect of the text adjusted according to the current frame and scheduling settings. Type should be `STRING`.
        - `NEG`: The output is a dynamically scheduled string, representing the negative aspect of the text adjusted according to the current frame and scheduling settings. Type should be `STRING`.
