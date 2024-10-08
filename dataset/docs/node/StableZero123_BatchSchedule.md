- `StableZero123_BatchSchedule`: The StableZero123_BatchSchedule node is designed to manage and schedule batch processing tasks for Stable Diffusion models, optimizing the workflow for generating images in batches. It focuses on efficiently organizing the rendering process to accommodate various frame counts and scheduling requirements, ensuring a streamlined operation for large-scale image generation projects.
    - Parameters:
        - `width`: Sets the width of the images to be generated, directly affecting the resolution and aspect ratio of the output. Type should be `INT`.
        - `height`: Specifies the height of the images to be generated, directly affecting the resolution and aspect ratio of the output. Type should be `INT`.
        - `batch_size`: Defines the number of images to be processed in a single batch, influencing the efficiency and speed of the batch processing task. Type should be `INT`.
        - `interpolation`: Determines the interpolation method used for processing images, affecting the smoothness and quality of transitions between frames. Type should be `COMBO[STRING]`.
        - `azimuth_points_string`: Specifies the azimuth conditions for 3D model rendering, influencing the orientation and angle of the generated images. Type should be `STRING`.
        - `elevation_points_string`: Defines the elevation conditions for 3D model rendering, affecting the vertical angle and perspective of the generated images. Type should be `STRING`.
    - Inputs:
        - `clip_vision`: Specifies the CLIP vision model to be used for conditioning the generation process, impacting the visual style and content of the generated images. Type should be `CLIP_VISION`.
        - `init_image`: Defines the initial image to start the batch processing from, setting the visual basis for subsequent image generations. Type should be `IMAGE`.
        - `vae`: Determines the variational autoencoder used for encoding and decoding images, crucial for the quality and characteristics of the output. Type should be `VAE`.
    - Outputs:
        - `positive`: Represents the positive conditioning output, influencing the generation towards desired attributes. Type should be `CONDITIONING`.
        - `negative`: Represents the negative conditioning output, used to steer the generation away from undesired attributes. Type should be `CONDITIONING`.
        - `latent`: Outputs the latent representation of the generated images, crucial for further processing or manipulation. Type should be `LATENT`.
