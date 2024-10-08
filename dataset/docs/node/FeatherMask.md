- `FeatherMask`: The FeatherMask node applies a feathering effect to the edges of a given mask, smoothly transitioning the mask's edges by adjusting their opacity based on specified distances from each edge. This creates a softer, more blended edge effect.
    - Parameters:
        - `left`: Specifies the distance from the left edge within which the feathering effect will be applied. Type should be `INT`.
        - `top`: Specifies the distance from the top edge within which the feathering effect will be applied. Type should be `INT`.
        - `right`: Specifies the distance from the right edge within which the feathering effect will be applied. Type should be `INT`.
        - `bottom`: Specifies the distance from the bottom edge within which the feathering effect will be applied. Type should be `INT`.
    - Inputs:
        - `mask`: The mask to which the feathering effect will be applied. It determines the area of the image that will be affected by the feathering. Type should be `MASK`.
    - Outputs:
        - `mask`: The output is a modified version of the input mask with a feathering effect applied to its edges. Type should be `MASK`.
