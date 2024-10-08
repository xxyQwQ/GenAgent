- `CLIPLoader`: The CLIPLoader node is designed for loading CLIP models, supporting different types such as stable diffusion and stable cascade. It abstracts the complexities of loading and configuring CLIP models for use in various applications, providing a streamlined way to access these models with specific configurations.
    - Parameters:
        - `clip_name`: Specifies the name of the CLIP model to be loaded. This name is used to locate the model file within a predefined directory structure. Type should be `COMBO[STRING]`.
        - `type`: Determines the type of CLIP model to load, offering options between 'stable_diffusion' and 'stable_cascade'. This affects how the model is initialized and configured. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `clip`: The loaded CLIP model, ready for use in downstream tasks or further processing. Type should be `CLIP`.
