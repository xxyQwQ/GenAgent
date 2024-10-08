- `Image Caption`: The Image Caption node is designed to add textual captions to images. It processes an image tensor, formats and wraps the provided text to fit the image dimensions, and then overlays the caption onto the image, optionally incorporating additional metadata into the image's PNG info.
    - Parameters:
        - `font`: Specifies the font to be used for the caption text, influencing the aesthetic presentation of the caption. Type should be `STRING`.
        - `caption`: The text content of the caption to be added to the image, which will be formatted and wrapped to fit within the image's dimensions. Type should be `STRING`.
    - Inputs:
        - `image`: The image tensor to which the caption will be added. It serves as the visual base for the caption overlay. Type should be `IMAGE`.
    - Outputs:
        - `image`: The modified image with the caption overlayed, potentially including additional PNG metadata. Type should be `IMAGE`.
