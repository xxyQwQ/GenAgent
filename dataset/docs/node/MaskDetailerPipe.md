- `MaskDetailerPipe`: The MaskDetailerPipe node is designed to enhance and refine mask details within images, leveraging advanced image processing techniques to improve the visual quality and accuracy of masks in various applications.
    - Parameters:
        - `guide_size`: Specifies the size of the guide for the detailing process, affecting the level of detail achievable. Type should be `FLOAT`.
        - `guide_size_for`: Specifies the guide size for the detailing process, affecting the precision and detail of the enhancement. Type should be `BOOLEAN`.
        - `max_size`: Defines the maximum size limit for the detailing process, ensuring the output stays within desired dimensions. Type should be `FLOAT`.
        - `mask_mode`: unknown Type should be `BOOLEAN`.
        - `seed`: unknown Type should be `INT`.
        - `steps`: unknown Type should be `INT`.
        - `cfg`: unknown Type should be `FLOAT`.
        - `sampler_name`: Specifies the sampler to use during the detailing process, affecting the quality of the output. Type should be `COMBO[STRING]`.
        - `scheduler`: Determines the scheduling algorithm for the detailing process, impacting the efficiency and outcome. Type should be `COMBO[STRING]`.
        - `denoise`: Flag to enable or disable denoising during the detailing process, affecting the clarity of the output. Type should be `FLOAT`.
        - `feather`: Feathering parameter to smooth edges in the detailing process, enhancing the visual quality of the mask. Type should be `INT`.
        - `crop_factor`: Factor by which to crop the image during the detailing process, affecting the focus area. Type should be `FLOAT`.
        - `drop_size`: Minimum size of details to be dropped during the detailing, affecting the level of detail retained. Type should be `INT`.
        - `refiner_ratio`: Ratio for refining the mask, affecting the intensity of refinement. Type should be `FLOAT`.
        - `batch_size`: The size of the batch for processing, affecting the throughput of the detailing process. Type should be `INT`.
        - `cycle`: Number of cycles to perform the detailing process, affecting the thoroughness of enhancement. Type should be `INT`.
        - `inpaint_model`: Flag to enable or disable the use of an inpainting model during the detailing process, affecting the handling of missing areas. Type should be `BOOLEAN`.
        - `noise_mask_feather`: Feathering parameter for the noise mask, affecting the blending of noise in the detailing. Type should be `INT`.
    - Inputs:
        - `image`: The 'image' parameter specifies the input image to be processed, serving as the primary subject for mask detailing and enhancement. Type should be `IMAGE`.
        - `mask`: The 'mask' parameter provides the initial mask to be refined, playing a key role in the detailing process by indicating areas of interest. Type should be `MASK`.
        - `basic_pipe`: The 'basic_pipe' parameter refers to a set of pre-configured models and settings used as a baseline for the detailing process, ensuring consistency and quality in output. Type should be `BASIC_PIPE`.
        - `refiner_basic_pipe_opt`: Optional parameter for an alternative set of models and settings for refining, offering customization. Type should be `BASIC_PIPE`.
        - `detailer_hook`: Hook for custom detailing logic, allowing for extended functionality. Type should be `DETAILER_HOOK`.
    - Outputs:
        - `image`: The refined or enhanced image after mask detailing, showcasing improved visual quality. Type should be `IMAGE`.
        - `cropped_refined`: A list of cropped and refined images, highlighting detailed areas of interest. Type should be `IMAGE`.
        - `cropped_enhanced_alpha`: A list of cropped images with enhanced alpha channels, providing transparency details. Type should be `IMAGE`.
        - `basic_pipe`: The set of models and settings used in the detailing process, returned for potential further use. Type should be `BASIC_PIPE`.
        - `refiner_basic_pipe_opt`: The optional set of alternative models and settings for refining, if used, returned for potential further use. Type should be `BASIC_PIPE`.