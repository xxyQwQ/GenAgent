- `LLMInputToDocuments`: The node is designed to transform input data into a structured document format, potentially incorporating additional information and supporting concatenation of inputs for enhanced document creation.
    - Parameters:
        - `extra_info`: Optional JSON-formatted string to include as metadata in the document, allowing for the enrichment of the document with additional context or information. Type should be `STRING`.
        - `concat_input`: A boolean flag indicating whether to concatenate input data into a single document or treat each input as a separate document, affecting the structure and granularity of the resulting documents. Type should be `BOOLEAN`.
    - Inputs:
        - `input_data`: Represents the primary data to be transformed into document format, serving as the core content for the document creation process. Type should be `*`.
    - Outputs:
        - `documents`: The transformed input data into a structured document or documents, enriched with optional metadata and adjusted according to the concatenation setting. Type should be `DOCUMENT`.
