- `easy imageSave`: The 'easy imageSave' node is designed to simplify the process of saving images. It abstracts the complexities involved in file handling and image encoding, providing a straightforward way for users to save images to disk with minimal configuration.
    - Parameters:
        - `filename_prefix`: Defines the prefix for the saved image filenames, allowing users to organize their saved images more effectively by categorizing them under a common prefix. Type should be `STRING`.
        - `only_preview`: Determines if the node should only preview the images without saving them, offering an option to review images before committing to save. Type should be `BOOLEAN`.
    - Inputs:
        - `images`: Specifies the images to be saved. This parameter is crucial as it directly influences the output by determining which images are processed and stored. Type should be `IMAGE`.
    - Outputs: