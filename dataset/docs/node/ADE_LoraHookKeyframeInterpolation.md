- `ADE_LoraHookKeyframeInterpolation`: This node is designed for creating interpolated LoRA hook keyframes, allowing for the dynamic adjustment of model behavior over time. It facilitates the generation of a sequence of keyframes based on specified start and end percentages, strengths, and interpolation method, enabling fine-grained control over the temporal evolution of model parameters.
    - Parameters:
        - `start_percent`: Defines the starting percentage for the interpolation, setting the initial point in the sequence of generated keyframes. Type should be `FLOAT`.
        - `end_percent`: Specifies the ending percentage for the interpolation, determining the final point in the sequence of generated keyframes. Type should be `FLOAT`.
        - `strength_start`: Sets the initial strength value for the interpolation, marking the beginning of the strength adjustment range. Type should be `FLOAT`.
        - `strength_end`: Determines the ending strength value for the interpolation, concluding the strength adjustment range. Type should be `FLOAT`.
        - `interpolation`: Selects the interpolation method to be used for generating the sequence of keyframes, influencing the transition between start and end values. Type should be `COMBO[STRING]`.
        - `intervals`: Specifies the number of intervals (or keyframes) to generate between the start and end points, affecting the granularity of the interpolation. Type should be `INT`.
        - `print_keyframes`: Optional. Controls whether the generated keyframes are logged, aiding in debugging and visualization of the interpolation process. Type should be `BOOLEAN`.
    - Inputs:
        - `prev_hook_kf`: Optional. Allows for the inclusion of a previously defined set of LoRA hook keyframes to which the new interpolated keyframes will be added. Type should be `LORA_HOOK_KEYFRAMES`.
    - Outputs:
        - `HOOK_KF`: Returns a group of LoRA hook keyframes, including both previously existing and newly interpolated keyframes, ready for application in model conditioning. Type should be `LORA_HOOK_KEYFRAMES`.
