- `WeightScheduleExtend`: The WeightScheduleExtend node is designed to extend, and convert if needed, different value lists/series. It supports various input types and can output the extended or converted values in the specified format, facilitating the manipulation and analysis of data within computational workflows.
    - Parameters:
        - `input_values_i`: unknown Type should be `FLOAT`.
        - `output_type`: Specifies the desired output format of the extended or converted data, allowing for flexibility in how the results are utilized or further processed. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `float`: The output is a float value or a collection of float values, depending on the operation performed and the output type specified. Type should be `FLOAT`.