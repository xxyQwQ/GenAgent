- `SaltAudioVisualizer`: The SaltAudioVisualizer node is designed to create visual representations of audio data. It processes audio input to generate visualizations that can be used for analysis or aesthetic purposes, highlighting the audio's structure, frequency content, and dynamics.
    - Parameters:
        - `frame_rate`: The 'frame_rate' parameter specifies the frame rate for the visualization, affecting the temporal resolution of the generated visual output. Type should be `INT`.
        - `start_frame`: The 'start_frame' parameter defines the starting point of the audio segment to be visualized, allowing for selective visualization of specific parts of the audio. Type should be `INT`.
        - `end_frame`: The 'end_frame' parameter determines the end point of the audio segment for visualization, enabling customization of the visualization's length. Type should be `INT`.
    - Inputs:
        - `audio`: The 'audio' parameter is the primary input for the visualization process, representing the audio data to be visualized. Type should be `AUDIO`.
    - Outputs: