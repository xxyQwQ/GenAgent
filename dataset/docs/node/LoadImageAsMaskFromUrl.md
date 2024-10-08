- `LoadImageAsMaskFromUrl`: This node is designed to load images from URLs and convert them into masks based on a specified color channel. It supports selecting from alpha, red, green, or blue channels to create the mask, facilitating various image processing and manipulation tasks where masks are required.
    - Parameters:
        - `url`: The URL(s) from which the image will be loaded. Supports loading multiple images if URLs are separated by new lines. This parameter is essential for fetching the image data to be processed into masks. Type should be `STRING`.
        - `channel`: Specifies the color channel ('alpha', 'red', 'green', 'blue') to be used for creating the mask. This choice determines which part of the image data will be converted into the mask, affecting the outcome significantly. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `masks`: The output masks generated from the images, based on the selected color channel. These masks are suitable for various image processing applications that require specific areas to be isolated or highlighted. Type should be `MASK`.
