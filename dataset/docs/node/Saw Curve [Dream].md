- `Saw Curve [Dream]`: The Saw Curve node generates a linear ramp waveform that cycles over a specified period, allowing for the creation of sawtooth wave animations based on frame count and time. It supports customization through parameters like maximum and minimum values, periodicity, and phase adjustment.
    - Parameters:
        - `max_value`: The maximum value the saw curve can reach within its cycle, defining the peak of the waveform. Type should be `FLOAT`.
        - `min_value`: The minimum value the saw curve can reach, defining the base of the waveform. Type should be `FLOAT`.
        - `periodicity_seconds`: The duration of one complete cycle of the saw curve in seconds, determining its frequency. Type should be `FLOAT`.
        - `phase`: A phase shift for the saw curve, allowing the waveform to be advanced or delayed within its cycle. Type should be `FLOAT`.
    - Inputs:
        - `frame_counter`: Represents the current frame count and time, used as the basis for calculating the saw curve's position within its cycle. Type should be `FRAME_COUNTER`.
    - Outputs:
        - `FLOAT`: The calculated float value of the saw curve at the current frame. Type should be `FLOAT`.
        - `INT`: An integer representation of the saw curve's current value, rounded from the float calculation. Type should be `INT`.