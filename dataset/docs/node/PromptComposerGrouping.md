- `PromptComposerGrouping`: The PromptComposerGrouping node is designed to modify and enhance input text based on specified weights and activity status, primarily focusing on grouping elements within prompts for AI-based applications.
    - Parameters:
        - `text_in`: The primary text input that serves as the base for grouping modifications. Type should be `STRING`.
        - `weight`: A numerical value that influences the degree of modification applied to the input text, with higher values indicating greater emphasis. Type should be `FLOAT`.
        - `active`: A boolean flag that determines whether the grouping modifications should be applied to the input text. Type should be `BOOLEAN`.
    - Inputs:
    - Outputs:
        - `text_out`: The modified text output after applying grouping logic, based on the input parameters. Type should be `STRING`.