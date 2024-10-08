- `Switch latent [Crystools]`: The `CSwitchBooleanLatent` node provides a mechanism to switch between two latent representations based on a boolean condition. It abstracts the decision-making process, allowing for dynamic selection of latent data streams.
    - Parameters:
        - `boolean`: The boolean condition that determines which latent representation (`on_true` or `on_false`) to return. It is central to the node's decision-making process. Type should be `BOOLEAN`.
    - Inputs:
        - `on_true`: The latent representation to be returned if the boolean condition is true. It plays a crucial role in determining the output based on the condition. Type should be `LATENT`.
        - `on_false`: The latent representation to be returned if the boolean condition is false. This parameter ensures an alternative output is available, enhancing the node's flexibility. Type should be `LATENT`.
    - Outputs:
        - `latent`: The selected latent representation based on the boolean condition. It encapsulates the node's core functionality of conditional selection. Type should be `LATENT`.
