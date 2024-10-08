- `XY Input Aesthetic Score`: This node is designed to generate aesthetic score values for XY plotting, allowing users to visualize and analyze the aesthetic quality of images or outputs based on specified criteria. It supports both positive and negative aesthetic scores, enabling a comprehensive assessment of aesthetic attributes.
    - Parameters:
        - `target_ascore`: Specifies the target aesthetic score type, either 'positive' or 'negative', to determine the nature of the aesthetic evaluation. Type should be `COMBO[STRING]`.
        - `batch_count`: Determines the number of aesthetic score values to generate, facilitating batch processing for efficiency and scalability. Type should be `INT`.
        - `first_ascore`: Sets the starting point of the aesthetic score range, enabling customization of the evaluation scope. Type should be `FLOAT`.
        - `last_ascore`: Defines the end point of the aesthetic score range, allowing for precise control over the evaluation interval. Type should be `FLOAT`.
    - Inputs:
    - Outputs:
        - `X or Y`: Outputs the generated aesthetic score values, categorized as either 'AScore+' for positive or 'AScore-' for negative, suitable for XY plotting. Type should be `XY`.
