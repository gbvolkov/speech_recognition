from pydub import AudioSegment
import os

def convert_m4a_to_wav(input_file, output_file, bitrate="192k"):
    """
    Converts an M4A file to MP3 format.

    :param input_file: Path to the input M4A file.
    :param output_file: Path where the output MP3 will be saved.
    :param bitrate: Bitrate for the MP3 file (default is 192k).
    """
    #try:
    # Load the M4A file
    audio = AudioSegment.from_file(input_file, format='m4a')


    # Export as MP3
    audio.export(output_file, format='wav')#, bitrate=bitrate)
    
    print(f"Successfully converted '{input_file}' to '{output_file}' with bitrate {bitrate}.")
    #except Exception as e:
    #    print(f"Error converting '{input_file}': {e}")

input_path = "audio/audio1097921934.m4a"   # Replace with your input file path
output_path = "audio/audio1097921934.wav" # Replace with your desired output file path
#convert_m4a_to_wav(input_path, output_path)
#fd = open(input_path, 'rb')

audios=["audio/audio1097921934.m4a", "audio/audio1415011527.m4a", "audio/audio1499365096.m4a"]

for audio in audios:
    out_audio, _ = os.path.splitext(audio)
    out_audio = f"{out_audio}.wav"
    convert_m4a_to_wav(audio, out_audio)


