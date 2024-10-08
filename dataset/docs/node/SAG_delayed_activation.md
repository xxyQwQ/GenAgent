- `SAG delayed activation`: This node introduces a custom implementation of Self-Attention Guidance (SAG) for neural networks, specifically designed to enhance model performance by dynamically adjusting attention mechanisms based on conditional inputs. It focuses on modifying the attention layers within a model to incorporate additional guidance, thereby potentially improving the model's ability to generate or process data with greater precision and relevance.
    - Parameters:
        - `scale`: A scaling factor that adjusts the intensity of the SAG effect, influencing how strongly the guidance affects the model's attention mechanisms. Type should be `FLOAT`.
        - `blur_sigma`: Specifies the standard deviation of the Gaussian blur applied for adversarial blurring, part of the SAG technique to manipulate attention scores. Type should be `FLOAT`.
        - `sigma_start`: The starting value of sigma for which the SAG modifications are applied, setting an operational range for the attention guidance. Type should be `FLOAT`.
        - `sigma_end`: The ending value of sigma, marking the lower bound of the operational range for applying the SAG modifications. Type should be `FLOAT`.
    - Inputs:
        - `model`: The neural network model to which the Self-Attention Guidance modifications will be applied. This parameter is crucial as it determines the base structure that will be enhanced with the custom SAG implementation. Type should be `MODEL`.
    - Outputs:
        - `model`: The modified neural network model, now enhanced with the custom Self-Attention Guidance implementation, aimed at improving its performance by dynamically adjusting its attention mechanisms. Type should be `MODEL`.
