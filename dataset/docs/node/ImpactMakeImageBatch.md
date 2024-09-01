- `ImpactMakeImageBatch`: The ImpactMakeImageBatch node is designed to aggregate multiple images into a single batch. This process involves potentially resizing images to ensure uniform dimensions across the batch, facilitating operations that require consistent image sizes. The node serves as a utility within the Impact Pack, streamlining the handling of images for batch processing.
    - Parameters:
    - Inputs:
        - `image1`: The primary image to which subsequent images will be concatenated. It serves as the reference for resizing operations if other images differ in dimensions. Type should be `IMAGE`.
    - Outputs:
        - `image`: A single tensor representing a batch of images, where each image has been resized as necessary to match the dimensions of the first image in the batch. Type should be `IMAGE`.