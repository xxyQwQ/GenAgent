- `ImageDrawChord`: The ImageDrawChord node is designed for drawing chords on images. It leverages geometric and color parameters to render chords, which are segments of the circumference of a circle, onto a specified image canvas, enhancing image customization and manipulation capabilities.
    - Parameters:
        - `width`: Specifies the width of the image canvas where the chord will be drawn, determining the horizontal dimension of the drawing area. Type should be `INT`.
        - `height`: Specifies the height of the image canvas where the chord will be drawn, determining the vertical dimension of the drawing area. Type should be `INT`.
        - `size`: Defines the thickness of the chord's outline, allowing for adjustable visual prominence within the image. Type should be `INT`.
        - `start_x`: The starting x-coordinate for the chord, marking one endpoint of the chord on the image canvas. Type should be `FLOAT`.
        - `start_y`: The starting y-coordinate for the chord, marking one endpoint of the chord on the image canvas. Type should be `FLOAT`.
        - `end_x`: The ending x-coordinate for the chord, marking the other endpoint of the chord on the image canvas. Type should be `FLOAT`.
        - `end_y`: The ending y-coordinate for the chord, marking the other endpoint of the chord on the image canvas. Type should be `FLOAT`.
        - `start`: The starting angle of the chord in degrees, defining the beginning of the arc segment. Type should be `INT`.
        - `end`: The ending angle of the chord in degrees, defining the end of the arc segment. Type should be `INT`.
        - `red`: The red color component of the chord, contributing to the chord's overall color. Type should be `INT`.
        - `green`: The green color component of the chord, contributing to the chord's overall color. Type should be `INT`.
        - `blue`: The blue color component of the chord, contributing to the chord's overall color. Type should be `INT`.
        - `alpha`: The alpha (transparency) component of the chord, allowing for adjustable opacity. Type should be `FLOAT`.
        - `SSAA`: Specifies the Super Sampling Anti-Aliasing factor, enhancing the chord's visual quality by reducing aliasing effects. Type should be `INT`.
        - `method`: Determines the resizing method used after drawing the chord, affecting the final image's visual quality. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `image`: The output is an image tensor with the drawn chord, showcasing the visual modification applied to the original image. Type should be `IMAGE`.
