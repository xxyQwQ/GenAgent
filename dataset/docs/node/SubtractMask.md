- `SubtractMask`: The SubtractMask node is designed to perform subtraction operations between two mask inputs, resulting in a single mask output that represents the difference between the two input masks. This operation is useful in scenarios where the removal of certain areas or features from a mask is required, effectively highlighting disparities or changes between the two masks.
    - Parameters:
    - Inputs:
        - `mask1`: The first mask input for the subtraction operation. It serves as the base mask from which the second mask will be subtracted. Type should be `MASK`.
        - `mask2`: The second mask input for the subtraction operation. This mask is subtracted from the first mask, effectively removing its features from the first mask. Type should be `MASK`.
    - Outputs:
        - `mask`: The resulting mask after subtracting the second mask from the first. This output highlights the differences or changes between the two input masks. Type should be `MASK`.
