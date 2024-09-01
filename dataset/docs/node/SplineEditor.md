- `SplineEditor`: The SplineEditor node is a graphical editor designed for creating and manipulating splines to generate various types of output data. It allows for intricate control over the spline's shape and characteristics through interactive editing features. This node is particularly useful for applications requiring custom schedules, mask batches, or coordinate transformations.
    - Parameters:
        - `points_store`: Stores the control points data, used for generating and manipulating the spline. Type should be `STRING`.
        - `coordinates`: A string representation of coordinates for control points, used to define the shape and trajectory of the spline. Type should be `STRING`.
        - `mask_width`: Specifies the width of the mask to be generated, affecting the spatial resolution of the output mask. Type should be `INT`.
        - `mask_height`: Defines the height of the mask, influencing the vertical resolution of the output mask. Type should be `INT`.
        - `points_to_sample`: Sets the number of sample points to generate from the spline, independent of the control points count. Type should be `INT`.
        - `sampling_method`: Chooses the sampling method, either along the time axis for schedules or along the path for coordinates. Type should be `COMBO[STRING]`.
        - `interpolation`: Specifies the method of interpolation between control points, impacting the smoothness and shape of the spline. Type should be `COMBO[STRING]`.
        - `tension`: Adjusts the tension of the spline, affecting its curvature and how tightly it adheres to the control points. Type should be `FLOAT`.
        - `repeat_output`: Determines how many times the output is repeated, useful for generating multiple instances of the output data. Type should be `INT`.
        - `float_output_type`: Determines the format of the floating-point output, allowing selection among list, pandas series, or tensor formats. Type should be `COMBO[STRING]`.
        - `min_value`: Specifies the minimum value for the output, setting a lower bound on the generated data. Type should be `FLOAT`.
        - `max_value`: Defines the maximum value for the output, establishing an upper limit on the generated data. Type should be `FLOAT`.
    - Inputs:
    - Outputs:
        - `mask`: Generates a mask batch based on the defined spline, useful for applications requiring mask inputs. Type should be `MASK`.
        - `coord_str`: Provides a string representation of coordinates derived from the spline, useful for textual representation of paths or shapes. Type should be `STRING`.
        - `float`: Outputs a list of floats, pandas series, or tensor, depending on the selected output type, representing sampled values from the spline. Type should be `FLOAT`.
        - `count`: Returns an integer count, potentially representing the number of elements in the output data. Type should be `INT`.