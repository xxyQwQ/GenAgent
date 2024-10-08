- `VHS_MergeLatents`: The VHS_MergeLatents node is designed for combining two sets of latents into a single set, applying various strategies for handling differences in dimensions and ensuring the merged set is suitable for further processing or generation tasks. It incorporates scaling and cropping methods to align the latents' dimensions before merging, making it a versatile tool in the manipulation and preparation of latent representations.
    - Parameters:
        - `merge_strategy`: Determines how the dimensions of the two latent sets are matched before merging, allowing for flexible handling of varying sizes. Type should be `COMBO[STRING]`.
        - `scale_method`: Specifies the method used for scaling latents to match dimensions, offering options like nearest-exact, bilinear, and bicubic among others. Type should be `COMBO[STRING]`.
        - `crop`: Defines the cropping method to be applied after scaling, if necessary, to ensure the dimensions are exactly aligned. Type should be `COMBO[STRING]`.
    - Inputs:
        - `latents_A`: The first set of latents to be merged. It plays a crucial role in the merging process, potentially serving as the template for dimension matching and scaling. Type should be `LATENT`.
        - `latents_B`: The second set of latents to be merged. Depending on the merge strategy, it may be scaled to match the dimensions of the first set before merging. Type should be `LATENT`.
    - Outputs:
        - `LATENT`: The merged set of latents, ready for further processing or generation tasks. Type should be `LATENT`.
        - `count`: The total number of latents in the merged set, providing a quick reference to the size of the output. Type should be `INT`.
