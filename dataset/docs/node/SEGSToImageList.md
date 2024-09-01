- `SEGSToImageList`: The SEGSToImageList node is designed to convert segmentation data into a list of images. It optionally adjusts the scale of segmentation data to match a fallback image and extracts cropped images from the segmentation data, providing a flexible way to handle segmentation outputs for further processing or visualization.
    - Parameters:
    - Inputs:
        - `segs`: The primary input containing segmentation data. It is essential for the operation as it holds the segments to be converted into images. Type should be `SEGS`.
        - `fallback_image_opt`: An optional image used to match the scale of segmentation data. If provided, it ensures that the segmentation data is appropriately scaled to align with the fallback image's dimensions. Type should be `IMAGE`.
    - Outputs:
        - `image`: A list of images extracted from the segmentation data. Each image corresponds to a cropped area from the original segmentation, potentially adjusted for scale. Type should be `IMAGE`.