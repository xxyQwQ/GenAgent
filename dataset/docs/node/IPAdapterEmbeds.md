- `IPAdapterEmbeds`: The IPAdapterEmbeds node is designed to handle embedding operations within the IPAdapter framework, focusing on the manipulation and processing of embeddings to enhance or modify the input data for further use in image processing or generation tasks.
    - Parameters:
        - `weight`: The 'weight' parameter allows for the adjustment of the influence of embeddings on the output, offering a means to fine-tune the generation process. Type should be `FLOAT`.
        - `weight_type`: The 'weight_type' parameter specifies the method of weighting to be applied to embeddings, affecting the overall impact on the generated output. Type should be `COMBO[STRING]`.
        - `start_at`: The 'start_at' parameter defines the starting point for embedding application, enabling precise control over the integration of embeddings into the generation process. Type should be `FLOAT`.
        - `end_at`: The 'end_at' parameter determines the endpoint for embedding application, allowing for targeted modifications to the generated output. Type should be `FLOAT`.
        - `embeds_scaling`: The 'embeds_scaling' parameter outlines the scaling strategy for embeddings, influencing how embeddings are adjusted and applied. Type should be `COMBO[STRING]`.
    - Inputs:
        - `model`: The 'model' parameter specifies the model to be used in conjunction with the IPAdapter, playing a pivotal role in how embeddings are applied or generated. Type should be `MODEL`.
        - `ipadapter`: The 'ipadapter' parameter indicates the specific IPAdapter instance to be used, crucial for determining the embedding manipulation or application strategy. Type should be `IPADAPTER`.
        - `pos_embed`: The 'pos_embed' parameter represents the positive embeddings to be processed, serving as a key input for operations aiming to enhance or modify image generation. Type should be `EMBEDS`.
        - `neg_embed`: The 'neg_embed' parameter represents the negative embeddings, providing a means to incorporate contrasting elements into the generation process. Type should be `EMBEDS`.
        - `attn_mask`: The 'attn_mask' parameter allows for the specification of an attention mask, offering additional control over the focus of embedding application. Type should be `MASK`.
        - `clip_vision`: The 'clip_vision' parameter indicates whether CLIP vision embeddings are to be used, potentially enhancing the relevance of generated outputs to textual descriptions. Type should be `CLIP_VISION`.
    - Outputs:
        - `model`: The output 'model' represents the modified or enhanced model after embedding operations, ready for further use in image generation tasks. Type should be `MODEL`.
