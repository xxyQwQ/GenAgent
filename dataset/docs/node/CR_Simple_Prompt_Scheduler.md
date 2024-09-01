- `CR Simple Prompt Scheduler`: The CR_SimplePromptScheduler node is designed to manage and schedule animation prompts based on a simple scheduling logic. It processes a list of keyframes and, depending on the scheduling mode and format, prepares a sequence of prompts for animation frames. This node supports different scheduling modes, including default prompts and keyframe lists, and can handle format conversions for compatibility with different animation systems.
    - Parameters:
        - `keyframe_list`: The list of keyframes to be scheduled. It is crucial for defining the sequence and timing of animation prompts. The format and content of this list directly influence the scheduling outcome and the generation of animation prompts. Type should be `STRING`.
        - `current_frame`: The current frame number for which the prompt is being scheduled. It determines the relevant prompt and keyframe information for the current point in the animation sequence. Type should be `INT`.
        - `keyframe_format`: Specifies the format of the keyframes, allowing for conversion between different animation systems' formats. This parameter ensures compatibility and correct interpretation of the keyframe list. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `current_prompt`: The prompt scheduled for the current frame, ready for use in generating the animation. Type should be `STRING`.
        - `next_prompt`: The prompt scheduled for the next frame, providing a lookahead for animation generation. Type should be `STRING`.
        - `weight`: A weight value indicating the relevance or influence of the current prompt in the context of animation blending or interpolation. Type should be `FLOAT`.
        - `show_help`: A URL to documentation or help resources related to the CR_SimplePromptScheduler node, offering guidance and additional information. Type should be `STRING`.