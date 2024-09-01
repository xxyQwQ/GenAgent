- `ADE_InjectI2VIntoAnimateDiffModel`: This node is designed to integrate the I2V (Image to Video) model into the AnimateDiff framework, enhancing its capabilities by incorporating motion models. It serves as a bridge to enrich AnimateDiff's animation process with additional motion dynamics, facilitating a more complex and nuanced animation output.
    - Parameters:
        - `model_name`: Specifies the name of the motion model to be loaded, playing a crucial role in determining the animation's motion dynamics. Type should be `COMBO[STRING]`.
    - Inputs:
        - `motion_model`: Represents the motion model object to be injected into the AnimateDiff model, crucial for applying specific motion dynamics to the animation process. Type should be `MOTION_MODEL_ADE`.
        - `ad_settings`: Optional settings for the AnimateDiff process, allowing for customization of the animation's appearance and behavior. Type should be `AD_SETTINGS`.
    - Outputs:
        - `MOTION_MODEL`: The enhanced AnimateDiff model with the injected I2V capabilities, ready for animation tasks. Type should be `MOTION_MODEL_ADE`.