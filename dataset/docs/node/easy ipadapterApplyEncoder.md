- `easy ipadapterApplyEncoder`: The node 'easy ipadapterApplyEncoder' is designed to encode images using an IPAdapter, producing both positive and negative embeddings. It allows for the customization of embedding generation through various parameters, enabling a tailored approach to image encoding within a given model's context.
    - Parameters:
        - `preset`: Defines the preset configuration to be used during the encoding process, influencing the characteristics of the generated embeddings. Type should be `COMBO[STRING]`.
        - `num_embeds`: Determines the number of embeddings to be generated, affecting the depth of the encoding process. Type should be `INT`.
        - `weight1`: Weight for the first image's influence on the embedding, allowing for customized emphasis. Type should be `FLOAT`.
        - `weight2`: Weight for the second image's embedding, customizable for balanced or biased emphasis. Type should be `FLOAT`.
        - `weight3`: Weight for the third image's embedding, enabling emphasis customization. Type should be `FLOAT`.
        - `combine_method`: Method to combine multiple embeddings, influencing the final embedding outcome. Type should be `COMBO[STRING]`.
    - Inputs:
        - `model`: Specifies the model to which the IPAdapter encoding process will be applied, serving as the foundation for embedding generation. Type should be `MODEL`.
        - `image1`: The primary image input for encoding, which is essential for generating the corresponding embeddings. Type should be `IMAGE`.
        - `image2`: The second image input for encoding, optional based on 'num_embeds', contributing to the diversity of generated embeddings. Type should be `IMAGE`.
        - `image3`: The third image input for encoding, optional based on 'num_embeds', further diversifying the embedding output. Type should be `IMAGE`.
        - `mask1`: Optional mask for the first image, guiding the focus of the encoding process. Type should be `MASK`.
        - `mask2`: Optional mask for the second image, if provided, to refine the encoding focus. Type should be `MASK`.
        - `mask3`: Optional mask for the third image, if provided, for further encoding refinement. Type should be `MASK`.
        - `optional_ipadapter`: An optional IPAdapter to be used, offering flexibility in the encoding process. Type should be `IPADAPTER`.
        - `pos_embeds`: Accumulated positive embeddings from the encoding process, reflecting the positive aspects of the images. Type should be `EMBEDS`.
        - `neg_embeds`: Accumulated negative embeddings from the encoding process, reflecting the negative aspects of the images. Type should be `EMBEDS`.
    - Outputs:
        - `model`: Returns the model after applying the IPAdapter encoding process, potentially modified with new embeddings. Type should be `MODEL`.
        - `ipadapter`: Provides the IPAdapter used in the encoding process, reflecting any adjustments made during embedding generation. Type should be `IPADAPTER`.
        - `pos_embed`: The combined positive embeddings resulting from the encoding process, ready for further application. Type should be `EMBEDS`.
        - `neg_embed`: The combined negative embeddings resulting from the encoding process, ready for further application. Type should be `EMBEDS`.