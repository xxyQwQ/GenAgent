- `Context Switch (rgthree)`: The Context Switch (rgthree) node is designed to streamline context management by selecting the first non-empty context from a set of provided contexts. This functionality ensures efficient determination of an active context for subsequent operations, facilitating smoother transitions and management within various processing flows.
    - Parameters:
    - Inputs:
        - `ctx_i`: Serves as a generic placeholder for any of the contexts provided to the node. The node evaluates each context in sequence until it finds the first non-empty one, which is then selected for output. This approach allows for flexible and dynamic context switching based on the availability of content within the contexts. Type should be `RGTHREE_CONTEXT`.
    - Outputs:
        - `CONTEXT`: The comprehensive context output, encompassing various aspects like model configuration, image processing parameters, and conditioning information, among others, based on the first non-empty context found. Type should be `RGTHREE_CONTEXT`.
        - `MODEL`: Outputs model-related context information. Type should be `MODEL`.
        - `CLIP`: Outputs CLIP model configuration context. Type should be `CLIP`.
        - `VAE`: Outputs VAE model configuration context. Type should be `VAE`.
        - `POSITIVE`: Outputs positive conditioning context. Type should be `CONDITIONING`.
        - `NEGATIVE`: Outputs negative conditioning context. Type should be `CONDITIONING`.
        - `LATENT`: Outputs latent space configuration context. Type should be `LATENT`.
        - `IMAGE`: Outputs image processing context. Type should be `IMAGE`.
        - `SEED`: Outputs seed value for random number generation context. Type should be `INT`.