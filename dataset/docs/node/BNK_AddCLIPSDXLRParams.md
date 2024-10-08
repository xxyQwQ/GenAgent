- `BNK_AddCLIPSDXLRParams`: This node is designed to enhance the conditioning data for image generation by incorporating additional parameters such as width, height, and an aesthetic score. It operates by iterating over a list of conditioning elements, modifying each with the specified dimensions and aesthetic score, thereby preparing the data for more tailored and aesthetically pleasing image generation.
    - Parameters:
        - `width`: Specifies the width to be added to the conditioning data, influencing the dimensions of the generated image. Type should be `INT`.
        - `height`: Specifies the height to be added to the conditioning data, influencing the dimensions of the generated image. Type should be `INT`.
        - `ascore`: An aesthetic score to be added to the conditioning data, aiming to guide the image generation towards more visually appealing results. Type should be `FLOAT`.
    - Inputs:
        - `conditioning`: The base conditioning data for image generation, which this node modifies by adding width, height, and an aesthetic score to each element. Type should be `CONDITIONING`.
    - Outputs:
        - `conditioning`: The enhanced conditioning data, now including specified width, height, and aesthetic score for each element. Type should be `CONDITIONING`.
