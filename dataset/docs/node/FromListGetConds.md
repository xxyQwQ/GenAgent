- `FromListGetConds`: This node is designed to retrieve a single conditioning element from a list based on a specified index. It enables random access to the list elements, including the ability to use negative indices for reverse access, thereby enhancing flexibility in handling conditioning data.
    - Parameters:
    - Inputs:
        - `list`: The list from which a conditioning element is to be retrieved. It is essential for specifying the source of data. Type should be `CONDITIONING`.
    - Outputs:
        - `conditioning`: The retrieved conditioning element at the specified index. Type should be `CONDITIONING`.