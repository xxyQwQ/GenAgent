- `Range(Num Steps) - Float`: This node generates a range of floating-point numbers based on specified start and stop values, and a number of steps. It allows for the creation of evenly spaced sequences of numbers, which can be used for various applications such as generating sample points or defining intervals.
    - Parameters:
        - `start`: Specifies the starting value of the range. It sets the initial point from which the sequence of floating-point numbers will begin. Type should be `FLOAT`.
        - `stop`: Defines the ending value of the range. It determines the final point up to which the sequence of floating-point numbers will extend. Type should be `FLOAT`.
        - `num_steps`: Determines the total number of steps or intervals within the specified range. This affects the spacing between each number in the sequence. Type should be `INT`.
    - Inputs:
    - Outputs:
        - `range`: A list of floating-point numbers representing the generated range based on the input parameters. Type should be `FLOAT`.
        - `range_sizes`: A list containing the size of each generated range, indicating how many numbers are in each sequence. Type should be `INT`.