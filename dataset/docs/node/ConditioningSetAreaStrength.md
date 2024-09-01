- `ConditioningSetAreaStrength`: This node specializes in adjusting the strength of conditioning applied to a specific area, allowing for fine-tuned control over the intensity of effects or modifications within that area.
    - Parameters:
        - `strength`: Specifies the intensity of the conditioning effect, enabling precise control over how strongly the area is influenced. Type should be `FLOAT`.
    - Inputs:
        - `conditioning`: The conditioning context to which the strength adjustment will be applied, serving as the foundation for the modification. Type should be `CONDITIONING`.
    - Outputs:
        - `conditioning`: Returns the modified conditioning context with the updated strength value applied. Type should be `CONDITIONING`.