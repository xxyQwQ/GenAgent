- `ImageRemoveBackground+`: This node is designed to remove the background from images, utilizing the rembg library to process each image in a batch and generate a corresponding mask for each image.
    - Parameters:
    - Inputs:
        - `rembg_session`: The session object for the rembg library, required for processing the images to remove their backgrounds. Type should be `REMBG_SESSION`.
        - `image`: The input image or batch of images to have their backgrounds removed. Type should be `IMAGE`.
    - Outputs:
        - `image`: The batch of images with their backgrounds removed. Type should be `IMAGE`.
        - `mask`: A mask or batch of masks corresponding to the removed backgrounds of the input images. Type should be `MASK`.
