- `SaltMaskHistogramEqualizationRegion`: This node applies histogram equalization to each mask in a collection of masks, enhancing the contrast of regions within the masks. It is designed to improve the visibility and differentiation of features within each mask by adjusting the distribution of intensities.
    - Parameters:
    - Inputs:
        - `masks`: The collection of masks to be processed. Each mask is enhanced individually to improve its contrast through histogram equalization. Type should be `MASK`.
    - Outputs:
        - `MASKS`: The enhanced masks with improved contrast, resulting from the application of histogram equalization to each original mask. Type should be `MASK`.