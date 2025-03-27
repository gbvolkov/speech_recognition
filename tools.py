import wave

def extract_wav_segment(input_file, output_file, start_time, duration):
    """
    Extracts a segment from a WAV file and writes it to a new file.
    
    Parameters:
        input_file (str): Path to the input WAV file.
        output_file (str): Path to save the extracted WAV segment.
        start_time (float): Start time in seconds for the segment.
        duration (float): Duration in seconds for the segment.
    """
    with wave.open(input_file, 'rb') as wav_in:
        # Retrieve audio parameters
        frame_rate = wav_in.getframerate()
        n_channels = wav_in.getnchannels()
        sampwidth = wav_in.getsampwidth()
        total_frames = wav_in.getnframes()

        # Calculate start and number of frames
        start_frame = int(start_time * frame_rate)
        num_frames = int(duration * frame_rate)
        
        # Check if the start_frame is within the file length
        if start_frame > total_frames:
            raise ValueError("Start time is beyond the end of the file.")
        
        # Adjust the number of frames if duration goes beyond the file length
        if start_frame + num_frames > total_frames:
            num_frames = total_frames - start_frame
        
        # Set the file's current position to the start frame and read frames
        wav_in.setpos(start_frame)
        frames = wav_in.readframes(num_frames)
    
    # Write the frames to the output file with the same parameters
    with wave.open(output_file, 'wb') as wav_out:
        wav_out.setnchannels(n_channels)
        wav_out.setsampwidth(sampwidth)
        wav_out.setframerate(frame_rate)
        wav_out.writeframes(frames)


if __name__ == "__main__":
    # Example usage:
    extract_wav_segment("./audio/audio1092719758.wav", "./audio/audio1092719758_segment1.wav", start_time=27, duration=20)

