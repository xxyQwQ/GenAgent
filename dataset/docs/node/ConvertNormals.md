- `ConvertNormals`: The `ConvertNormals` node is designed to transform normal maps between different formats, adjust their scale, and optionally normalize or fill missing areas. It supports various input and output modes, allowing for flexible adaptation of normal maps to suit different rendering or processing requirements. This node is particularly useful in graphics and image processing pipelines where normal maps need to be converted or adjusted for specific uses.
    - Parameters:
        - `input_mode`: Specifies the format of the input normal map, influencing how it is initially processed. This parameter determines the initial adjustments made to the normal map before further transformations. Type should be `COMBO[STRING]`.
        - `output_mode`: Defines the desired format for the output normal map, guiding the final adjustments made to the tensor. This parameter ensures the transformed normal map meets specific format requirements. Type should be `COMBO[STRING]`.
        - `scale_XY`: A scaling factor applied to the X and Y components of the normal map, affecting its overall appearance. This parameter allows for fine-tuning the visual impact of the normal map on surfaces. Type should be `FLOAT`.
        - `normalize`: A boolean flag indicating whether the normal map should be normalized after transformations. Normalization ensures the normal map's vectors are unit length, often required for consistent lighting effects. Type should be `BOOLEAN`.
        - `fix_black`: A boolean flag that, when enabled, applies a correction to black areas in the normal map, potentially filling these areas based on the `optional_fill` parameter. This is useful for repairing or enhancing normal maps with missing data. Type should be `BOOLEAN`.
    - Inputs:
        - `normals`: The input normal map tensor to be transformed. This parameter is central to the node's operation, as it undergoes various transformations based on the other input parameters. Type should be `IMAGE`.
        - `optional_fill`: An optional tensor used to fill black areas in the normal map if `fix_black` is enabled. This parameter allows for custom fill patterns or colors to be applied, enhancing the versatility of the node. Type should be `IMAGE`.
    - Outputs:
        - `image`: The transformed normal map tensor, adjusted according to the specified input and output modes, scaling, normalization, and optional filling. This output is ready for use in subsequent processing or rendering steps. Type should be `IMAGE`.