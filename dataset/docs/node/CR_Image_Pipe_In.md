- `CR Image Pipe In`: This node is designed to initialize and configure an image processing pipeline, allowing for the specification of image properties such as dimensions and upscale factor. It serves as the entry point for images into a customizable processing flow, facilitating subsequent manipulations or analyses.
    - Parameters:
        - `width`: Specifies the desired width of the image. It affects the dimensions of the image as it moves through the pipeline. Type should be `INT`.
        - `height`: Specifies the desired height of the image. It affects the dimensions of the image as it moves through the pipeline. Type should be `INT`.
        - `upscale_factor`: Determines the factor by which the image should be upscaled. This parameter influences the resolution enhancement of the image. Type should be `FLOAT`.
    - Inputs:
        - `image`: The initial image to be processed. It sets the starting point for the pipeline. Type should be `IMAGE`.
    - Outputs:
        - `pipe`: A pipeline configuration encapsulating the image and its specified properties. Type should be `PIPE_LINE`.
        - `show_help`: A URL providing detailed documentation and help for using the CR Image Pipe In node. Type should be `STRING`.