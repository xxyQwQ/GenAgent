- `APersonMaskGenerator`: The APersonMaskGenerator node is designed to generate segmented masks for different parts of a person in an image, such as hair, face, body, and clothes. It utilizes image segmentation techniques to identify and isolate these areas, creating masks that can be used for various image editing and processing tasks.
    - Parameters:
        - `face_mask`: Indicates whether a mask for the face should be generated, affecting the segmentation process by isolating the face area. Type should be `BOOLEAN`.
        - `background_mask`: Indicates whether a mask for the background should be generated, affecting the segmentation process by isolating areas not covered by other specified masks. Type should be `BOOLEAN`.
        - `hair_mask`: Indicates whether a mask for the hair should be generated, guiding the segmentation to isolate the hair area. Type should be `BOOLEAN`.
        - `body_mask`: Indicates whether a mask for the body should be generated, guiding the segmentation to isolate the body area. Type should be `BOOLEAN`.
        - `clothes_mask`: Indicates whether a mask for the clothes should be generated, guiding the segmentation to isolate the clothes area. Type should be `BOOLEAN`.
        - `confidence`: Specifies the confidence threshold for mask generation, affecting the precision of the segmentation and the resulting masks. Type should be `FLOAT`.
    - Inputs:
        - `images`: The input images for which the masks are to be generated. They are crucial as they serve as the base for all segmentation operations, determining the areas to be isolated and masked. Type should be `IMAGE`.
    - Outputs:
        - `masks`: The output is a collection of masks for the specified targets, each representing a segmented area of the image. These masks can be used for further image editing or processing tasks. Type should be `MASK`.