- `CR Color Panel`: The CR_ColorPanel node is designed to generate a simple color panel image with customizable dimensions and fill color. It allows for the creation of a solid color background that can be used in various graphical layouts or as a base for further graphical manipulation.
    - Parameters:
        - `panel_width`: Specifies the width of the color panel. The width influences the size of the generated image, allowing for customization according to the user's needs. Type should be `INT`.
        - `panel_height`: Determines the height of the color panel. Similar to the width, it affects the overall size of the output image, providing flexibility in the panel's dimensions. Type should be `INT`.
        - `fill_color`: Defines the primary color used to fill the panel. This parameter is crucial for setting the visual appearance of the color panel. Type should be `COMBO[STRING]`.
        - `fill_color_hex`: An optional hexadecimal color code that can override the primary fill color, offering an alternative method for specifying the panel's color. Type should be `STRING`.
    - Inputs:
    - Outputs:
        - `image`: The generated color panel as an image, ready for use in various graphical contexts. Type should be `IMAGE`.
        - `show_help`: A URL providing additional help and documentation related to the CR_ColorPanel node. Type should be `STRING`.