- `LLMChatEngine`: The LLMChatEngine node facilitates interactive chat sessions using a language model, enabling users to input queries and receive text responses. It manages the instantiation and resetting of a chat engine based on user input, ensuring dynamic and contextually relevant interactions.
    - Parameters:
        - `query`: The user's input query as a string, which is processed by the chat engine to generate a relevant text response. This input is essential for driving the conversation forward. Type should be `STRING`.
        - `reset_engine`: A boolean flag indicating whether to reset the chat engine before processing the current query, allowing for fresh interactions without prior context. Type should be `BOOLEAN`.
    - Inputs:
        - `llm_index`: Represents the index of the language learning model to be used for the chat session, crucial for initializing or resetting the chat engine to ensure responses are generated accurately and contextually. Type should be `LLM_INDEX`.
    - Outputs:
        - `string`: The text response generated by the chat engine in response to the user's query. Type should be `STRING`.
