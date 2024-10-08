- `Draw Contour(s)`: The DrawContours node is designed for visualizing contours on images by drawing them over the original image. It allows for customization of the contour visualization, such as selecting specific contours to draw, adjusting the thickness of the contour lines, and choosing their color, thereby enhancing the interpretability of contour-based analyses.
    - Parameters:
        - `index_to_draw`: Specifies which contour from the collection to draw. A value of -1 indicates that all contours should be drawn. Type should be `INT`.
        - `thickness`: The thickness of the contour lines. A negative value indicates that contours should be filled. Type should be `INT`.
    - Inputs:
        - `image`: The original image on which contours are to be drawn. It serves as the background for contour visualization. Type should be `IMAGE`.
        - `contours`: A collection of contours to be drawn on the image. Each contour is a sequence of points defining its shape. Type should be `CV_CONTOURS`.
        - `color`: The color of the contour lines. This allows for customization of the visual appearance of contours. Type should be `COLOR`.
    - Outputs:
        - `image`: The image with the specified contours drawn over it, enhancing visual analysis of contours. Type should be `IMAGE`.
