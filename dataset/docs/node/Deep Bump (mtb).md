- `Deep Bump (mtb)`: The Deep Bump node is designed for generating normal and height maps from single images, offering a versatile approach to texture processing by converting color images to normal maps, normal maps to curvature maps, or normal maps to height maps, depending on the selected mode. It utilizes advanced image processing techniques to achieve detailed and accurate representations of surface textures.
    - Parameters:
        - `mode`: Specifies the operation mode of the node, which can be converting color images to normal maps, normal maps to curvature maps, or normal maps to height maps, affecting the type of texture map produced. Type should be `COMBO[STRING]`.
        - `color_to_normals_overlap`: Determines the overlap size when converting color images to normal maps, influencing the smoothness and continuity of the generated normal map. Type should be `COMBO[STRING]`.
        - `normals_to_curvature_blur_radius`: Specifies the blur radius when converting normal maps to curvature maps, affecting the level of detail and smoothness in the curvature map. Type should be `COMBO[STRING]`.
        - `normals_to_height_seamless`: A boolean indicating whether the conversion from normal maps to height maps should be seamless, impacting the continuity and uniformity of the height map. Type should be `BOOLEAN`.
    - Inputs:
        - `image`: The input image for which the normal or height map will be generated. It serves as the foundational data from which the node derives texture maps. Type should be `IMAGE`.
    - Outputs:
        - `image`: The output is either a normal map, curvature map, or height map based on the selected mode, representing the processed texture of the input image. Type should be `IMAGE`.