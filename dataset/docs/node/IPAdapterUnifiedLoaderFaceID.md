- `IPAdapterUnifiedLoaderFaceID`: This node is designed to unify the loading process for FaceID models, accommodating various presets and configurations to ensure compatibility and optimal performance across different model types and computational environments.
    - Parameters:
        - `preset`: Defines the specific FaceID preset to be used, allowing for customization and optimization based on the model's intended use case. Type should be `COMBO[STRING]`.
        - `lora_strength`: Adjusts the strength of the LoRA (Low-Rank Adaptation) adjustments, providing fine-tuning capabilities for the model's performance. Type should be `FLOAT`.
        - `provider`: Determines the computational backend for model execution, supporting a range of environments from CPU to various GPU architectures. Type should be `COMBO[STRING]`.
    - Inputs:
        - `model`: Specifies the model to be loaded, serving as a key identifier for selecting the appropriate FaceID model configuration. Type should be `MODEL`.
        - `ipadapter`: Optionally specifies an IPAdapter model to be used in conjunction with the FaceID model, enhancing its capabilities. Type should be `IPADAPTER`.
    - Outputs:
        - `MODEL`: The loaded model, ready for use with the specified configurations and enhancements. Type should be `MODEL`.
        - `ipadapter`: The optionally specified IPAdapter model, loaded and configured for use. Type should be `IPADAPTER`.