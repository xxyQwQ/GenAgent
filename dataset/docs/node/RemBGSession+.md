- `RemBGSession+`: The RemBGSession+ node is designed to create a new session for background removal tasks, supporting a variety of models and execution providers. It abstracts the complexity of initializing a session with specific models and providers, facilitating the removal of backgrounds from images with flexibility and efficiency.
    - Parameters:
        - `model`: Specifies the model to be used for background removal, offering a selection from general-purpose to specialized models for human segmentation and cloth parsing, among others. This choice directly influences the accuracy and performance of the background removal process. Type should be `COMBO[STRING]`.
        - `providers`: Determines the execution provider for the background removal task, allowing for selection among various hardware acceleration options like CPU, CUDA, and more. This choice affects the performance and compatibility of the background removal operation. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `rembg_session`: Represents a session initialized for background removal tasks, ready to be used with images for removing backgrounds. Type should be `REMBG_SESSION`.