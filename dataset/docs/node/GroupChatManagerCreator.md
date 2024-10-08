- `GroupChatManagerCreator`: This node is designed to facilitate the creation of a group chat manager, capable of overseeing and managing the interactions within a group chat environment. It specializes in configuring and initializing a chat manager agent that is tailored for group chat scenarios, ensuring smooth communication and interaction management among multiple participants.
    - Parameters:
        - `name`: The name of the group chat manager, serving as an identifier and a label for the chat management entity. Type should be `STRING`.
        - `system_message`: A system message that can be used by the group chat manager, typically for announcements or instructions within the group chat. Type should be `STRING`.
        - `max_consecutive_auto_reply`: An optional parameter that limits the maximum number of consecutive automatic replies by the chat manager, helping to prevent spam and maintain conversation quality. Type should be `INT`.
    - Inputs:
        - `llm_model`: An optional parameter specifying the large language model (LLM) configuration for the chat manager, enabling advanced language understanding and response generation capabilities. Type should be `LLM_MODEL`.
    - Outputs:
        - `group_manager`: The configured group chat manager, ready to be utilized in managing a group chat session. Type should be `GROUP_MANAGER`.
