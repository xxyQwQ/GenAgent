- `Inference_Core_LayeredDiffusionDecode`: This node specializes in decoding layered diffusion processes, enabling the transformation of sampled data into images through a series of diffusion steps. It leverages advanced techniques to efficiently handle and decode multiple layers of diffusion, optimizing the generation of high-quality images.
    - Parameters:
        - `sd_version`: Specifies the version of the diffusion model to be used, affecting the decoding behavior and output quality. Type should be `COMBO[STRING]`.
        - `sub_batch_size`: The size of sub-batches for processing, optimizing computational efficiency and resource usage. Type should be `INT`.
    - Inputs:
        - `samples`: The sampled data to be decoded into images, representing the initial input for the diffusion process. Type should be `LATENT`.
        - `images`: A tensor of images to be processed through the diffusion steps, serving as the basis for the decoding operation. Type should be `IMAGE`.
    - Outputs:
        - `image`: The images generated from the decoded diffusion process, representing the primary output. Type should be `IMAGE`.
        - `mask`: The alpha mask associated with the decoded images, providing transparency information. Type should be `MASK`.