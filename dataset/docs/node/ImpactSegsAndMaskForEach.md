- `ImpactSegsAndMaskForEach`: This node applies a mask to each segment within a collection of segments, performing a bitwise AND operation between the mask and the segment's cropped mask. It's designed to process each segment individually, allowing for precise control over how masks are applied to segmented parts of an image.
    - Parameters:
    - Inputs:
        - `segs`: The collection of segments to which the masks will be applied. Each segment represents a part of an image that has been isolated based on certain criteria. Type should be `SEGS`.
        - `masks`: A collection of masks to be applied to each corresponding segment. Each mask defines areas to be kept or removed in the segment's cropped mask through a bitwise AND operation. Type should be `MASK`.
    - Outputs:
        - `segs`: The modified collection of segments after applying the masks, where each segment's cropped mask has been updated based on the bitwise AND operation with the corresponding mask. Type should be `SEGS`.