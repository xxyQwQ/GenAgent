- `CR Apply ControlNet`: This node applies a ControlNet to an image based on specified conditions, allowing for dynamic image manipulation through control networks. It optionally provides a mechanism to adjust the intensity of the effect and to enable or disable the application of the ControlNet.
    - Parameters:
        - `switch`: A toggle to enable or disable the application of the ControlNet, allowing for conditional application. Type should be `COMBO[STRING]`.
        - `strength`: Adjusts the intensity of the ControlNet's effect on the image, providing control over the manipulation's strength. Type should be `FLOAT`.
    - Inputs:
        - `conditioning`: Represents the conditions under which the ControlNet is applied, affecting how the image is manipulated. Type should be `CONDITIONING`.
        - `control_net`: The ControlNet to be applied to the image, defining the specific transformations or effects. Type should be `CONTROL_NET`.
        - `image`: The image to which the ControlNet is applied, serving as the base for manipulation. Type should be `IMAGE`.
    - Outputs:
        - `CONDITIONING`: The modified conditions after applying the ControlNet. Type should be `CONDITIONING`.
        - `show_help`: A URL to documentation or help related to the node's functionality. Type should be `STRING`.