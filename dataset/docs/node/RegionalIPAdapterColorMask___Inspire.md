- `RegionalIPAdapterColorMask __Inspire`: The RegionalIPAdapterColorMask node is designed to apply regional image processing adaptations based on color masks. It enables the integration of specific image embeddings and adjustments within designated areas of an image, identified by color, to achieve localized image modification or enhancement.
    - Parameters:
        - `mask_color`: Defines the color used to identify the region of interest within the image for adaptations. Type should be `STRING`.
        - `weight`: Determines the intensity or influence of the applied embeddings on the specified region. Type should be `FLOAT`.
        - `noise`: Specifies the level of noise to be applied in conjunction with the embeddings for the adaptation effect. Type should be `FLOAT`.
        - `weight_type`: Specifies the method of applying weight to the embeddings, offering options like original, linear, or channel penalty for flexibility in adaptation. Type should be `COMBO[STRING]`.
        - `start_at`: Marks the beginning of the effect application within the adaptation process, allowing for phased or gradual implementations. Type should be `FLOAT`.
        - `end_at`: Defines the endpoint for the effect application, enabling precise control over the extent of adaptations. Type should be `FLOAT`.
        - `unfold_batch`: A boolean flag that, when set, allows for batch processing of images, enhancing efficiency in adaptations. Type should be `BOOLEAN`.
        - `faceid_v2`: An optional boolean flag to enable or disable face identification version 2 for more refined adaptations. Type should be `BOOLEAN`.
        - `weight_v2`: An optional weight parameter for version 2 adaptations, providing additional control over the adaptation intensity. Type should be `FLOAT`.
        - `combine_embeds`: Specifies the method for combining embeddings, with options like concat, add, subtract, average, and norm average, offering versatility in effect application. Type should be `COMBO[STRING]`.
    - Inputs:
        - `color_mask`: Specifies the image to which the color mask will be applied, serving as the basis for regional adaptations. Type should be `IMAGE`.
        - `image`: The target image for which the adaptations are intended, providing a context for the applied effects. Type should be `IMAGE`.
        - `neg_image`: An optional negative image that can be used to specify undesired effects or adjustments within the region, offering a counterbalance to the primary image. Type should be `IMAGE`.
    - Outputs:
        - `regional_ipadapter`: The adapted image processing settings, encapsulating the regional adaptations based on the specified color mask and embeddings. Type should be `REGIONAL_IPADAPTER`.
        - `mask`: The generated mask based on the specified color, identifying the region of interest for adaptations. Type should be `MASK`.
