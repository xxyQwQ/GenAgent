- `OneButtonPreset`: The OneButtonPreset node is designed to streamline the process of applying preset configurations to a prompt generation task. It allows for the selection and application of predefined or custom settings that adjust various aspects of the prompt generation, such as theme, complexity, and style, enhancing the user's ability to produce tailored content with minimal effort.
    - Parameters:
        - `OneButtonPreset`: Specifies the preset configuration to be applied. This can be a predefined preset or a custom configuration, influencing the generation process by setting themes, styles, and complexity levels. Type should be `COMBO[STRING]`.
        - `base_model`: Defines the underlying model to be used for prompt generation, affecting the style and quality of the generated content. Type should be `COMBO[STRING]`.
        - `prompt_enhancer`: An optional component that modifies the generated prompt to meet specific criteria or add creative elements, further customizing the output. Type should be `COMBO[STRING]`.
        - `seed`: Determines the random seed used for generating prompts, ensuring reproducibility or variability in the output. Type should be `INT`.
    - Inputs:
    - Outputs:
        - `prompt`: The generated prompt based on the selected preset configuration, incorporating any specified enhancements or adjustments. Type should be `STRING`.