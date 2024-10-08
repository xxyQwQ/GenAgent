- `BboxVisualize`: The BboxVisualize node is designed to overlay bounding boxes on images, enhancing visual analysis by clearly demarcating areas of interest with specified line widths and colors.
    - Parameters:
        - `line_width`: Specifies the thickness of the lines used to draw the bounding boxes, affecting the visibility and prominence of the highlighted areas. Type should be `INT`.
    - Inputs:
        - `images`: A batch of images on which bounding boxes will be drawn. The images serve as the canvas for the visualization process. Type should be `IMAGE`.
        - `bboxes`: A list of bounding box coordinates specifying the areas to be highlighted on the images. These coordinates play a crucial role in determining the exact regions to be visualized. Type should be `BBOX`.
    - Outputs:
        - `images`: The modified batch of images with bounding boxes drawn over them, ready for visual inspection or further processing. Type should be `IMAGE`.
