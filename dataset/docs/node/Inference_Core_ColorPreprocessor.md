- `Inference_Core_ColorPreprocessor`: The Color Preprocessor node is designed to analyze and process images to detect and adjust their color palette. It utilizes a color detection algorithm to enhance or modify the image's color properties based on the specified resolution.
    - Parameters:
        - `resolution`: Specifies the resolution at which the color detection and adjustment should be performed, affecting the precision and quality of the output. Type should be `INT`.
    - Inputs:
        - `image`: The input image to be processed for color detection and adjustment. Type should be `IMAGE`.
    - Outputs:
        - `image`: The processed image with adjusted color properties, based on the color detection algorithm. Type should be `IMAGE`.
