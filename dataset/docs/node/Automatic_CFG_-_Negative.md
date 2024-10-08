- `Automatic CFG - Negative`: This node specializes in applying a dynamic configuration to models for generating content, with a focus on enhancing the generation process by adjusting the model's behavior based on negative prompts. It leverages an advanced configuration technique to fine-tune the model's output, aiming to mitigate the influence of negative aspects specified by the user.
    - Parameters:
        - `boost`: The boost parameter determines whether to skip unconditional generation steps, effectively altering the model's generation process to focus more on the specified conditions. Type should be `BOOLEAN`.
        - `negative_strength`: This parameter controls the strength of the negative conditioning, allowing for fine-tuning how strongly the model should mitigate or ignore the specified negative aspects during generation. Type should be `FLOAT`.
    - Inputs:
        - `model`: The model parameter is the core component that the node modifies, applying a dynamic configuration to adjust its behavior for content generation. Type should be `MODEL`.
    - Outputs:
        - `model`: The modified model with applied dynamic configuration, tailored to enhance content generation by considering negative prompts. Type should be `MODEL`.
