- `ImageDrawPolygon`: The ImageDrawPolygon node is designed for drawing regular polygons on an image canvas. It allows for customization of the polygon's size, number of sides, rotation, outline, and fill color, including alpha transparency. This node leverages supersampling for higher quality rendering and supports different resizing methods to adjust the final image size.
    - Parameters:
        - `size`: Specifies the size of the polygon to be drawn, affecting both its height and width. Type should be `INT`.
        - `sides`: Determines the number of sides of the regular polygon, defining its shape. Type should be `INT`.
        - `rotation`: Sets the rotation angle of the polygon in degrees, allowing for orientation adjustments. Type should be `INT`.
        - `outline_size`: Defines the thickness of the polygon's outline. Type should be `INT`.
        - `outline_red`: Specifies the red component of the outline color, part of the RGBA color model. Type should be `INT`.
        - `outline_green`: Specifies the green component of the outline color, part of the RGBA color model. Type should be `INT`.
        - `outline_blue`: Specifies the blue component of the outline color, part of the RGBA color model. Type should be `INT`.
        - `outline_alpha`: Determines the alpha transparency of the outline, allowing for semi-transparent effects. Type should be `FLOAT`.
        - `fill_red`: Specifies the red component of the fill color, part of the RGBA color model. Type should be `INT`.
        - `fill_green`: Specifies the green component of the fill color, part of the RGBA color model. Type should be `INT`.
        - `fill_blue`: Specifies the blue component of the fill color, part of the RGBA color model. Type should be `INT`.
        - `fill_alpha`: Determines the alpha transparency of the fill, allowing for semi-transparent effects. Type should be `FLOAT`.
        - `SSAA`: Sets the supersampling anti-aliasing factor for higher quality rendering. Type should be `INT`.
        - `method`: Chooses the method for resizing the image after drawing the polygon, affecting the final image quality. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `image`: Outputs the image tensor with the drawn polygon, ready for further processing or visualization. Type should be `IMAGE`.
