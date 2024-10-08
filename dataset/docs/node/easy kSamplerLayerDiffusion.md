- `easy kSamplerLayerDiffusion`: The `easy kSamplerLayerDiffusion` node is designed to integrate layer diffusion techniques into the sampling process, enhancing image generation with more control over the blending and detailing of generated images. It leverages various diffusion methods to apply nuanced modifications to images, supporting both foreground and background blending, attention mechanisms, and convolutional approaches for a refined output.
    - Parameters:
        - `image_output`: Specifies the desired output format for the generated images, influencing how the final images are presented or saved. Type should be `COMBO[STRING]`.
        - `link_id`: A unique identifier used to link the current diffusion process with other processes or data within the pipeline. Type should be `INT`.
        - `save_prefix`: Defines the prefix for filenames when saving generated images, allowing for organized storage and retrieval. Type should be `STRING`.
    - Inputs:
        - `pipe`: Represents the pipeline configuration, including model and sampling settings, crucial for the layer diffusion process. Type should be `PIPE_LINE`.
        - `model`: The model used for the diffusion process, central to determining the characteristics and quality of the generated images. Type should be `MODEL`.
    - Outputs:
        - `pipe`: The modified pipeline configuration after applying layer diffusion, reflecting changes in image blending, sampling, and additional settings. Type should be `PIPE_LINE`.
        - `final_image`: The final image result after the layer diffusion process, showcasing the applied blending and detailing effects. Type should be `IMAGE`.
        - `original_image`: The original image before the application of layer diffusion, allowing for comparison with the final result. Type should be `IMAGE`.
        - `alpha`: A value representing the blending factor used in the diffusion process, indicating the degree of blending between the original and diffused elements. Type should be `MASK`.
