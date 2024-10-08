- `IFUnet VFI`: The IFUnet_VFI node is designed for video frame interpolation, leveraging deep learning models to predict and generate intermediate frames between existing frames in a video sequence. This process enhances video smoothness and can be used for various applications such as slow-motion video generation, video restoration, and improving video frame rates.
    - Parameters:
        - `ckpt_name`: Specifies the checkpoint name for the model to be used in the interpolation process, determining the specific pre-trained weights and configuration. Type should be `COMBO[STRING]`.
        - `clear_cache_after_n_frames`: Controls how often the cache is cleared during the interpolation process to manage memory usage effectively. Type should be `INT`.
        - `multiplier`: Determines the number of intermediate frames to be generated between each pair of original frames, directly affecting the smoothness of the output video. Type should be `INT`.
        - `scale_factor`: A factor that scales the resolution of the output frames, allowing for adjustments in the size of the interpolated frames. Type should be `FLOAT`.
        - `ensemble`: A boolean flag indicating whether to use ensemble methods for interpolation, potentially improving the quality of the output frames. Type should be `BOOLEAN`.
    - Inputs:
        - `frames`: The input video frames to be interpolated, provided as a tensor. This is the core data on which the interpolation model operates. Type should be `IMAGE`.
        - `optional_interpolation_states`: Provides the option to specify states for selective frame interpolation, enabling more control over which frames are processed. Type should be `INTERPOLATION_STATES`.
    - Outputs:
        - `image`: The output interpolated video frames, enhancing the smoothness and temporal resolution of the input video sequence. Type should be `IMAGE`.
