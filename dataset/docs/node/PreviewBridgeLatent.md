- `PreviewBridgeLatent`: The PreviewBridgeLatent node is designed to facilitate the previewing of latent representations by decoding them into visual formats. It leverages various decoding strategies and optional VAE configurations to convert latent tensors into images, supporting different preview methods tailored to specific latent formats.
    - Parameters:
        - `image`: unknown Type should be `STRING`.
        - `preview_method`: Determines the decoding strategy and the format of the resulting image. This input is essential for selecting the appropriate method to decode the latent representation into its visual counterpart. Type should be `COMBO[STRING]`.
    - Inputs:
        - `latent`: The latent representation to be decoded into an image. This input is crucial as it contains the encoded information of the image in a compressed form, which is then transformed into a visual format based on the specified preview method and optional VAE configuration. Type should be `LATENT`.
        - `vae_opt`: An optional parameter that allows for the customization of the decoding process with specific VAE models and settings. Providing a VAE configuration can significantly influence the decoding outcome by utilizing tailored models. Type should be `VAE`.
    - Outputs:
        - `latent`: unknown Type should be `LATENT`.
        - `mask`: An optional output that represents a mask applied to the decoded image, used for further processing or visualization purposes. Type should be `MASK`.
