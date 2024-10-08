- `AIO_Preprocessor`: The AIO_Preprocessor node is designed to dynamically select and apply a specified auxiliary preprocessing operation on an image, based on the preprocessor type chosen. It supports a variety of preprocessing options, automatically configuring and executing the appropriate auxiliary preprocessor to modify the image according to the selected preprocessor's requirements.
    - Parameters:
        - `preprocessor`: Specifies the type of preprocessing to apply to the image. This selection determines which auxiliary preprocessor's logic will be executed, impacting the final preprocessing outcome on the image. Type should be `COMBO[STRING]`.
        - `resolution`: The resolution for the preprocessing operation, which may be used by certain preprocessors to adjust the processing detail level or output resolution. Type should be `INT`.
    - Inputs:
        - `image`: The input image to be preprocessed. This image is directly passed to the selected auxiliary preprocessor for modification. Type should be `IMAGE`.
    - Outputs:
        - `image`: The preprocessed image, as modified by the selected auxiliary preprocessor. Type should be `IMAGE`.
