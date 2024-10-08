- `ImageCompositeRelativeByContainer`: This node is designed for creating composite images by positioning and merging two sets of images relative to a container's dimensions. It dynamically calculates the placement of images based on the container's size and the specified relative positions, ensuring the images are appropriately scaled and positioned before merging them according to a specified method.
    - Parameters:
        - `images_a_x`: The relative horizontal position (as a percentage) for the first set of images within the container. Type should be `FLOAT`.
        - `images_a_y`: The relative vertical position (as a percentage) for the first set of images within the container. Type should be `FLOAT`.
        - `images_b_x`: The relative horizontal position (as a percentage) for the second set of images within the container. Type should be `FLOAT`.
        - `images_b_y`: The relative vertical position (as a percentage) for the second set of images within the container. Type should be `FLOAT`.
        - `background`: Specifies which set of images (either 'images_a' or 'images_b') should be treated as the background in the final composite image. Type should be `COMBO[STRING]`.
        - `method`: The method used for compositing the images, which can affect the appearance of the merged result. Type should be `COMBO[STRING]`.
    - Inputs:
        - `container`: The container image that serves as a reference for scaling and positioning the other images. Its dimensions dictate how the other images are adjusted and placed. Type should be `IMAGE`.
        - `images_a`: The first set of images to be composited. These images are adjusted and positioned relative to the container's dimensions. Type should be `IMAGE`.
        - `images_b`: The second set of images to be composited. These images are also adjusted and positioned relative to the container's dimensions, similar to the first set. Type should be `IMAGE`.
    - Outputs:
        - `image`: The final composite image resulting from the merging and positioning of the two sets of images relative to the container's dimensions. Type should be `IMAGE`.
