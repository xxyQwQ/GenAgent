- `CR Module Output`: The CR Module Output node is designed to serve as the endpoint for a module within a pipeline, facilitating the organized output of data processed through the module. It abstracts the complexity of data handling at the end of a module, ensuring a streamlined and efficient data flow out of the module.
    - Parameters:
        - `seed`: unknown Type should be `INT`.
    - Inputs:
        - `pipe`: Accepts a pipeline object that encapsulates all the data processed within the module. This object is the primary input for the node, enabling the organized output of processed data. Type should be `PIPE_LINE`.
        - `model`: unknown Type should be `MODEL`.
        - `pos`: unknown Type should be `CONDITIONING`.
        - `neg`: unknown Type should be `CONDITIONING`.
        - `latent`: unknown Type should be `LATENT`.
        - `vae`: unknown Type should be `VAE`.
        - `clip`: unknown Type should be `CLIP`.
        - `controlnet`: unknown Type should be `CONTROL_NET`.
        - `image`: unknown Type should be `IMAGE`.
    - Outputs:
        - `pipe`: Outputs the same pipeline object that was input, potentially with modifications or additions made during the module's processing. Type should be `PIPE_LINE`.
        - `show_help`: Provides a URL to the documentation or help page related to this node, offering users guidance on its usage and functionalities. Type should be `STRING`.