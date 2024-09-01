- `CM_Vec2UnaryOperation`: The node performs unary operations on 2-dimensional vectors, transforming a single vector based on a specified operation.
    - Parameters:
        - `op`: Specifies the unary operation to be performed on the vector, affecting the transformation result. Type should be `COMBO[STRING]`.
    - Inputs:
        - `a`: The 2-dimensional vector to be transformed by the unary operation. Type should be `VEC2`.
    - Outputs:
        - `vec2`: The transformed 2-dimensional vector resulting from the specified unary operation. Type should be `VEC2`.