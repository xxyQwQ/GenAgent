- `ImpactLatentInfo`: The ImpactLatentInfo node is designed to process latent representations, specifically by analyzing their shape and dimensions. It abstracts the complexity of handling latent spaces by providing a straightforward interface for extracting critical dimensional information.
    - Parameters:
    - Inputs:
        - `value`: The 'value' parameter represents the latent representation to be analyzed. It is crucial for determining the shape and dimensions of the latent space, which are essential for further processing or manipulation. Type should be `LATENT`.
    - Outputs:
        - `batch`: Represents the batch size of the input latent representation. Type should be `INT`.
        - `height`: Indicates the modified height dimension of the latent representation. Type should be `INT`.
        - `width`: Indicates the modified width dimension of the latent representation. Type should be `INT`.
        - `channel`: Represents the channel dimension of the input latent representation. Type should be `INT`.