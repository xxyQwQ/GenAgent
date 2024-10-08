- `VHS_SelectEveryNthMask`: This node is designed to streamline the process of selecting every Nth mask from a batch of masks, facilitating operations such as thinning out data or creating subsets for specific processing needs. It abstracts the complexity of batch manipulation, offering a straightforward way to reduce the volume of mask data by periodic sampling.
    - Parameters:
        - `select_every_nth`: Specifies the interval at which masks are selected from the input batch. This parameter defines the thinning rate, playing a pivotal role in the output by determining the frequency of mask selection within the batch. Type should be `INT`.
    - Inputs:
        - `mask`: The input mask tensor from which every Nth mask will be selected. This parameter is crucial for determining the subset of masks to be processed, directly impacting the node's output by filtering the input data based on the specified interval. Type should be `MASK`.
    - Outputs:
        - `MASK`: The output tensor containing every Nth mask selected from the input batch, effectively reducing the dataset size based on the specified interval. Type should be `MASK`.
        - `count`: The total count of masks selected and returned by the node, providing a straightforward way to understand the output's volume. Type should be `INT`.
