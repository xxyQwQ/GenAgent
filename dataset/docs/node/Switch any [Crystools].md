- `Switch any [Crystools]`: This node provides a mechanism to switch between two values based on a boolean condition. It abstracts the conditional logic, allowing for a clean and straightforward way to choose between two possible outcomes.
    - Parameters:
        - `boolean`: The boolean condition that determines which of the two values ('on_true' or 'on_false') to return. It is the core of the switch functionality, enabling dynamic decision-making. Type should be `BOOLEAN`.
    - Inputs:
        - `on_true`: The value to return if the boolean condition evaluates to true. It plays a crucial role in determining the node's output based on the condition. Type should be `*`.
        - `on_false`: The value to return if the boolean condition evaluates to false. This parameter ensures that an alternative outcome is available, making the switch operation complete. Type should be `*`.
    - Outputs:
        - `*`: unknown Type should be `*`.
