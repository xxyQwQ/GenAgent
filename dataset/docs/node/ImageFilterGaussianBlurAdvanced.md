- `ImageFilterGaussianBlurAdvanced`: This node applies an advanced Gaussian blur filter to images, allowing for separate horizontal and vertical blur sizes and standard deviations. It enhances image processing capabilities by providing more control over the blurring effect.
    - Parameters:
        - `size_x`: Specifies the horizontal size of the Gaussian kernel. It influences the extent of blurring along the x-axis. Type should be `INT`.
        - `size_y`: Specifies the vertical size of the Gaussian kernel. It influences the extent of blurring along the y-axis. Type should be `INT`.
        - `sigma_x`: Determines the horizontal standard deviation of the Gaussian kernel. It affects the spread of the blur along the x-axis. Type should be `INT`.
        - `sigma_y`: Determines the vertical standard deviation of the Gaussian kernel. It affects the spread of the blur along the y-axis. Type should be `INT`.
    - Inputs:
        - `images`: The images to be processed. This parameter is crucial for defining the input on which the Gaussian blur will be applied. Type should be `IMAGE`.
    - Outputs:
        - `image`: The output is the blurred image, processed using the specified parameters for the Gaussian blur. Type should be `IMAGE`.
