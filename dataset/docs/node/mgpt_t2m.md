- `mgpt_t2m`: The node transforms textual descriptions into motion data sequences using a MotionGPT model, enabling the generation of motion sequences based on natural language inputs.
    - Parameters:
        - `motion_length`: Specifies the desired length of the generated motion sequence. It influences the granularity and extent of the motion details. Type should be `INT`.
        - `seed`: A seed for random number generation, ensuring reproducibility of the motion sequences generated from the same inputs. Type should be `INT`.
        - `text`: The textual description that guides the generation of the motion sequence, serving as the creative input for the motion synthesis. Type should be `STRING`.
    - Inputs:
        - `mgpt_model`: The MotionGPT model used for generating motion sequences from text. It's crucial for interpreting the textual input and producing corresponding motion data. Type should be `MGPTMODEL`.
    - Outputs:
        - `motion_data`: The generated motion data sequence, represented as joint positions over time, derived from the textual description. Type should be `MOTION_DATA`.