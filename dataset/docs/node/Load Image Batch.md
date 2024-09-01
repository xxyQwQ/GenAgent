- `Load Image Batch`: The Load Image Batch node is designed to aggregate multiple images into a single batch tensor. It dynamically combines input images based on provided parameters, facilitating operations that require batch processing of images.
    - Parameters:
        - `mode`: Specifies the loading mode, which can be single or multiple images, affecting how images are aggregated into the batch. Type should be `COMBO[STRING]`.
        - `index`: Determines the starting index for image loading, enabling partial or staggered batch processing. Type should be `INT`.
        - `label`: Assigns a label to the loaded batch, useful for identification and tracking purposes. Type should be `STRING`.
        - `path`: Specifies the directory path where images are located, serving as the source for batch loading. Type should be `STRING`.
        - `pattern`: Defines the pattern used to match filenames within the specified path, allowing for selective loading of images. Type should be `STRING`.
        - `allow_RGBA_output`: Controls whether RGBA images are allowed in the output batch, accommodating images with transparency. Type should be `COMBO[STRING]`.
        - `filename_text_extension`: Determines whether the filename extension is included in the output, affecting the representation of image names. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `image`: The aggregated batch of images, represented as a tensor, ready for further processing or analysis. Type should be `IMAGE`.
        - `filename_text`: The text representation of filenames included in the batch, useful for tracking and identification of individual images. Type should be `STRING`.