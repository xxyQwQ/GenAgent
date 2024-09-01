- `ImageFilterMode`: The ImageFilterMode node applies a mode filter to images, which replaces each pixel's value with the most frequent value of its neighbors within a specified size, enhancing uniformity or reducing noise.
    - Parameters:
        - `size`: Determines the size of the neighborhood around each pixel to consider for calculating the mode, affecting the extent of filtering. Type should be `INT`.
    - Inputs:
        - `images`: Specifies the images to be processed, serving as the primary input for the mode filtering operation. Type should be `IMAGE`.
    - Outputs:
        - `image`: Returns the images after applying the mode filter, showcasing enhanced uniformity or reduced noise. Type should be `IMAGE`.