- `ImageGridComposite3x3`: This node is designed to concatenate nine input images into a single 3x3 grid, effectively creating a composite image that showcases all inputs in a structured layout.
    - Parameters:
    - Inputs:
        - `image1`: The first image to be placed in the top-left corner of the 3x3 grid. Type should be `IMAGE`.
        - `image2`: The second image to be placed in the top row of the 3x3 grid, next to the first image. Type should be `IMAGE`.
        - `image3`: The third image to be placed in the top row of the 3x3 grid, next to the second image. Type should be `IMAGE`.
        - `image4`: The fourth image to be placed in the middle row of the 3x3 grid, starting from the left. Type should be `IMAGE`.
        - `image5`: The central image in the 3x3 grid, surrounded by the other eight images. Type should be `IMAGE`.
        - `image6`: The sixth image to be placed in the middle row of the 3x3 grid, next to the fifth image. Type should be `IMAGE`.
        - `image7`: The seventh image to be placed in the bottom row of the 3x3 grid, starting from the left. Type should be `IMAGE`.
        - `image8`: The eighth image to be placed in the bottom row of the 3x3 grid, next to the seventh image. Type should be `IMAGE`.
        - `image9`: The ninth and final image to be placed in the bottom right corner of the 3x3 grid, completing the composite. Type should be `IMAGE`.
    - Outputs:
        - `image`: The composite image formed by concatenating the nine input images into a 3x3 grid. Type should be `IMAGE`.