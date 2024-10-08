- `Style Conditioner`: The StyleConditioner node is designed to apply specific stylistic adjustments to conditioning bases and refiners, leveraging a combination of predefined and dynamically generated style prompts. It utilizes seed-based selection for styles and performs encoding and averaging operations to blend the selected style with the existing conditioning, enabling nuanced control over the stylistic output.
    - Parameters:
        - `style`: Specifies the style to be applied. This can dynamically change the appearance or thematic elements of the output by adjusting the conditioning bases and refiners according to the selected style. Type should be `COMBO[STRING]`.
        - `strength`: Determines the intensity of the style application, affecting how strongly the selected style influences the conditioning bases and refiners. Type should be `FLOAT`.
        - `use_seed`: Indicates whether a seed should be used for selecting the style, enabling deterministic style selection. Type should be `COMBO[STRING]`.
        - `seed`: The seed value used for deterministic style selection when 'use_seed' is true. Type should be `INT`.
    - Inputs:
        - `positive_cond_base`: The initial positive conditioning base to which the style adjustments will be applied. Type should be `CONDITIONING`.
        - `negative_cond_base`: The initial negative conditioning base to which the style adjustments will be applied. Type should be `CONDITIONING`.
        - `positive_cond_refiner`: The initial positive conditioning refiner to which the style adjustments will be applied. Type should be `CONDITIONING`.
        - `negative_cond_refiner`: The initial negative conditioning refiner to which the style adjustments will be applied. Type should be `CONDITIONING`.
        - `base_clip`: The CLIP model used for encoding the base conditioning. Type should be `CLIP`.
        - `refiner_clip`: The CLIP model used for encoding the refiner conditioning. Type should be `CLIP`.
    - Outputs:
        - `base_pos_cond`: Returns the updated positive conditioning base after style adjustments. Type should be `CONDITIONING`.
        - `base_neg_cond`: Returns the updated negative conditioning base after style adjustments. Type should be `CONDITIONING`.
        - `refiner_pos_cond`: Returns the updated positive conditioning refiner after style adjustments. Type should be `CONDITIONING`.
        - `refiner_neg_cond`: Returns the updated negative conditioning refiner after style adjustments. Type should be `CONDITIONING`.
        - `style_str`: Returns the string representation of the applied style. Type should be `STRING`.
