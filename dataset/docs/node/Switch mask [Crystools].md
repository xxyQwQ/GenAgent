- `Switch mask [Crystools]`: The 'Switch mask [Crystools]' node allows for conditional selection between two mask inputs based on a boolean value. It serves as a control structure to dynamically choose one of two paths in data flow, based on the provided boolean condition.
    - Parameters:
        - `boolean`: The boolean condition that determines which mask (on_true or on_false) to return. It acts as the switch for selecting the output path. Type should be `BOOLEAN`.
    - Inputs:
        - `on_true`: The mask to be returned if the boolean condition is true. It plays a crucial role in determining the output based on the condition. Type should be `MASK`.
        - `on_false`: The mask to be returned if the boolean condition is false. This input provides an alternative path for the data flow, depending on the boolean condition. Type should be `MASK`.
    - Outputs:
        - `mask`: The selected mask output, determined by the boolean condition. It represents the conditional choice between the on_true and on_false inputs. Type should be `MASK`.
