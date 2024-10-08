- `DetailerForEach`: The DetailerForEach node is designed to iterate over a collection of items, applying a detailed analysis or transformation to each item individually. This process enhances the granularity of the analysis or transformation, ensuring that each item is processed with a focus on its specific characteristics or requirements.
    - Parameters:
        - `guide_size`: unknown Type should be `FLOAT`.
        - `guide_size_for`: unknown Type should be `BOOLEAN`.
        - `max_size`: unknown Type should be `FLOAT`.
        - `seed`: unknown Type should be `INT`.
        - `steps`: unknown Type should be `INT`.
        - `cfg`: unknown Type should be `FLOAT`.
        - `sampler_name`: unknown Type should be `COMBO[STRING]`.
        - `scheduler`: unknown Type should be `COMBO[STRING]`.
        - `denoise`: unknown Type should be `FLOAT`.
        - `feather`: unknown Type should be `INT`.
        - `noise_mask`: unknown Type should be `BOOLEAN`.
        - `force_inpaint`: unknown Type should be `BOOLEAN`.
        - `wildcard`: unknown Type should be `STRING`.
        - `cycle`: unknown Type should be `INT`.
        - `inpaint_model`: unknown Type should be `BOOLEAN`.
        - `noise_mask_feather`: unknown Type should be `INT`.
    - Inputs:
        - `image`: The 'image' input type is essential for operations that involve visual data, allowing the node to apply transformations or analyses directly to images. Type should be `IMAGE`.
        - `segs`: unknown Type should be `SEGS`.
        - `model`: The 'model' input type specifies the computational model used for processing, which is crucial for defining the behavior and capabilities of the node. Type should be `MODEL`.
        - `clip`: The 'clip' input type is used for operations that involve CLIP models, enabling the node to leverage textual or visual embeddings for analysis or transformation. Type should be `CLIP`.
        - `vae`: The 'vae' input type indicates the use of a Variational Autoencoder, important for tasks involving latent space manipulations or generative processes. Type should be `VAE`.
        - `positive`: The 'positive' input type represents conditioning information with a positive connotation, influencing the direction of the transformation or analysis. Type should be `CONDITIONING`.
        - `negative`: The 'negative' input type signifies conditioning information with a negative connotation, affecting the node's processing to account for undesired aspects. Type should be `CONDITIONING`.
        - `detailer_hook`: unknown Type should be `DETAILER_HOOK`.
    - Outputs:
        - `image`: This output type provides the transformed or analyzed images, reflecting the changes or insights gained through the node's processing. Type should be `IMAGE`.
