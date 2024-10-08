- `SaltListLinearInterpolation`: The SaltListLinearInterpolation node is designed to perform linear interpolation between two lists of schedule values based on a specified interpolation factor. This node is essential for creating smooth transitions between different states or values in a schedule, allowing for the generation of intermediate states that blend the characteristics of the input schedules.
    - Parameters:
        - `interpolation_factor`: A floating-point value between 0.0 and 1.0 that determines the weight of each input list in the interpolated output. A factor of 0.0 yields the first list, while 1.0 yields the second list. Type should be `FLOAT`.
    - Inputs:
        - `schedule_list_a`: The first list of schedule values to interpolate from. It serves as the starting point for the interpolation process. Type should be `LIST`.
        - `schedule_list_b`: The second list of schedule values to interpolate towards. It acts as the target endpoint for the interpolation process. Type should be `LIST`.
    - Outputs:
        - `schedule_list`: The resulting list of interpolated schedule values, blending the input lists according to the interpolation factor. Type should be `LIST`.
