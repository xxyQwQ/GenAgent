- `ImageDuplicator`: The ImageDuplicator node is designed to duplicate each image in a given batch, effectively increasing the number of images by replicating them. This node serves the purpose of augmenting the dataset or preparing the data for processes that require multiple instances of the same image.
    - Parameters:
        - `dup_times`: Specifies the number of times each image should be duplicated. This parameter controls the extent of duplication, thereby determining the total number of images produced by the node. Type should be `INT`.
    - Inputs:
        - `images`: The images to be duplicated. This parameter is crucial as it directly influences the node's operation by specifying which images are to be processed and duplicated. Type should be `IMAGE`.
    - Outputs:
        - `image`: The output of the node, consisting of the original images along with their duplicates, effectively increasing the total number of images. Type should be `IMAGE`.
