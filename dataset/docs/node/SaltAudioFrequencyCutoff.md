- `SaltAudioFrequencyCutoff`: This node applies a frequency cutoff to an audio file, using FFmpeg to filter out frequencies above or below a specified cutoff point. It's designed to modify the audio's frequency spectrum, either by attenuating frequencies beyond a certain threshold or enhancing the audio within a specific frequency range.
    - Parameters:
        - `filter_type`: Specifies the type of filter to apply (e.g., lowpass or highpass), determining whether frequencies above or below the cutoff frequency are attenuated. Type should be `COMBO[STRING]`.
        - `cutoff_frequency`: The frequency threshold for the filter. Frequencies above or below this value will be attenuated, depending on the filter type. Type should be `INT`.
    - Inputs:
        - `audio`: The raw audio data to be processed. This parameter is crucial as it represents the audio content that will undergo the frequency cutoff operation. Type should be `AUDIO`.
    - Outputs:
        - `audio`: The modified audio data after applying the frequency cutoff. This output reflects the changes made to the audio's frequency spectrum. Type should be `AUDIO`.