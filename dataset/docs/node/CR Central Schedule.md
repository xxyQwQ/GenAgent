- `CR Central Schedule`: This node serves as a central hub for managing and coordinating multiple animation schedules within a project. It allows for the integration and synchronization of various animation schedules, ensuring a cohesive animation flow across different elements or scenes.
    - Parameters:
        - `schedule_1`: Defines the first animation schedule to be managed, allowing for the specification of complex animation sequences. Type should be `STRING`.
        - `schedule_type1`: Specifies the type of the first animation schedule, categorizing it for appropriate processing and integration. Type should be `COMBO[STRING]`.
        - `schedule_alias1`: Provides an alias for the first animation schedule, facilitating easier reference and management within the node. Type should be `STRING`.
        - `schedule_2`: Defines the second animation schedule to be managed, extending the node's capability to coordinate multiple schedules. Type should be `STRING`.
        - `schedule_type2`: Specifies the type of the second animation schedule, enhancing the node's ability to categorize and process various schedules. Type should be `COMBO[STRING]`.
        - `schedule_alias2`: Provides an alias for the second animation schedule, aiding in its identification and management. Type should be `STRING`.
        - `schedule_3`: Defines the third animation schedule to be managed, further broadening the node's capacity to handle multiple animation sequences. Type should be `STRING`.
        - `schedule_type3`: Specifies the type of the third animation schedule, allowing for its proper categorization and integration. Type should be `COMBO[STRING]`.
        - `schedule_alias3`: Provides an alias for the third animation schedule, simplifying its reference and management. Type should be `STRING`.
        - `schedule_format`: Determines the format in which the schedules are to be processed, supporting different animation frameworks. Type should be `COMBO[STRING]`.
    - Inputs:
        - `schedule`: An optional schedule parameter that can be used to provide additional scheduling information. Type should be `SCHEDULE`.
    - Outputs:
        - `SCHEDULE`: The combined or processed animation schedule resulting from the node's operation, ready for use in the animation project. Type should be `SCHEDULE`.
        - `show_text`: Provides textual information or feedback about the processed schedule, aiding in debugging or further adjustments. Type should be `STRING`.
