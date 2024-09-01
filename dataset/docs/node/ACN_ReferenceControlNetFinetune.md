- `ACN_ReferenceControlNetFinetune`: This node specializes in refining the application of reference-based control mechanisms within a neural network, focusing on enhancing the integration of attention and adaptive instance normalization (AdaIN) techniques for improved style fidelity and reference weighting. It aims to fine-tune the control network's response to reference inputs, ensuring a more precise and effective adaptation to the provided references.
    - Parameters:
        - `attn_style_fidelity`: Specifies the fidelity of the style to be maintained when applying attention mechanisms, influencing the network's ability to preserve the stylistic aspects of the reference. Type should be `FLOAT`.
        - `attn_ref_weight`: Determines the weight of the reference input in the attention mechanism, affecting how strongly the reference influences the output. Type should be `FLOAT`.
        - `attn_strength`: Controls the overall strength of the attention mechanism, adjusting the impact of the reference on the network's output. Type should be `FLOAT`.
        - `adain_style_fidelity`: Defines the fidelity of the style to be maintained when applying adaptive instance normalization (AdaIN), influencing the preservation of stylistic elements from the reference. Type should be `FLOAT`.
        - `adain_ref_weight`: Sets the weight of the reference input in the AdaIN mechanism, modifying the extent to which the reference affects the output. Type should be `FLOAT`.
        - `adain_strength`: Adjusts the strength of the AdaIN mechanism, determining the influence of the reference on the final output. Type should be `FLOAT`.
    - Inputs:
    - Outputs:
        - `control_net`: The refined control network, enhanced for better integration and application of reference-based control mechanisms like attention and AdaIN. Type should be `CONTROL_NET`.