- `AV_LLMMessage`: The AV_LLMMessage node is designed for creating and managing messages within a language model chat interface. It supports the inclusion of text and optionally images, facilitating rich, interactive dialogues between users and the system or assistant. This node plays a crucial role in structuring conversations, ensuring messages are correctly formatted and adhere to specified roles.
    - Parameters:
        - `role`: Specifies the role of the message, such as 'system', 'user', or 'assistant', influencing how the message is processed and presented within the chat flow. Type should be `COMBO[STRING]`.
        - `text`: The main content of the message, which can be a user query, system response, or assistant's reply. Supports multiline text, allowing for detailed and comprehensive messages. Type should be `STRING`.
    - Inputs:
        - `image`: An optional image to accompany the text message, enhancing the interaction with visual content. The image is expected to be in Tensor format and converted to base64 for inclusion. Type should be `IMAGE`.
        - `messages`: A list of existing messages to which the new message will be added. This allows for the accumulation and management of conversation history. Type should be `LLM_MESSAGE`.
    - Outputs:
        - `messages`: Returns a list of messages, including the newly created message, facilitating the continuation and tracking of the conversation flow. Type should be `LLM_MESSAGE`.