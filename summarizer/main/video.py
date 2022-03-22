import torch
from pydub.utils import make_chunks
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from tkinter import *
from tkinter import filedialog
import moviepy.editor as mp
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa


nltk.download("punkt")
# Loading the model and the tokenizer
model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

filename = ""
transcript = []
fo = ""

def browseFiles():
    global filename
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(
                                              [("all files",
                                                "*.*")]))
    # Change label contents
    filename = filename.replace('/', '\\')
    print(filename)


def load_data(input_file):
    """ Function for resampling to ensure that the speech input is sampled at 16KHz.
    """
    # read the file
    speech, sample_rate = librosa.load(input_file)
    # make it 1-D
    if len(speech.shape) > 1:
        speech = speech[:, 0] + speech[:, 1]
    # Resampling at 16KHz since wav2vec2-base-960h is pretrained and fine-tuned on speech audio sampled at 16 KHz.
    if sample_rate != 16000:
        speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)
    return speech


def correct_casing(input_sentence):
    """ This function is for correcting the casing of the generated transcribed text
    """
    sentences = nltk.sent_tokenize(input_sentence)
    return (' '.join([s.replace(s[0], s[0].capitalize(), 1) for s in sentences]))


def asr_transcript(input_file):
    """This function generates transcripts for the provided audio input
    """
    speech = load_data(input_file)
    # Tokenize
    input_values = tokenizer(speech, return_tensors="pt").input_values
    # Take logits
    logits = model(input_values).logits
    # Take argmax
    predicted_ids = torch.argmax(logits, dim=-1)
    # Get the words from predicted word ids
    transcription = tokenizer.decode(predicted_ids[0])
    # Output is all upper case
    transcription = correct_casing(transcription.lower())
    return transcription


def convert_video_to_audio_moviepy(video_file, output_ext="wav"):
    filename, ext = os.path.splitext(video_file)
    clip = mp.VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}")





# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)


def splitFunction(audio):
    global transcript
    myaudio = AudioSegment.from_file(audio, "wav")

    # Split track where the silence is 2 seconds or more and get chunks using
    # the imported function.
    chunks = split_on_silence(
        # Use the loaded audio.
        myaudio,
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len=2000,
        silence_thresh=-50
    )

    # Process each chunk with your parameters
    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=500)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

        chunk_name = "chunk{0}.wav".format(i + 1)
        # Export the audio chunk with new bitrate.

        normalized_chunk.export(
            chunk_name,
            bitrate="192k",
            format="wav"
        )
        length = librosa.get_duration(filename=chunk_name)
        if (length < 2):
            os.remove(chunk_name)
        elif (length > 60):
            chunk_length_ms = 60000
            myaudio = AudioSegment.from_file(chunk_name, "wav")
            chunks = make_chunks(myaudio, chunk_length_ms)
            for i, chunk in enumerate(chunks):
                chunk_name_sub = "{0}.wav".format(i + 1)
                chunk.export(chunk_name_sub, format="wav")
                length_sub = librosa.get_duration(filename=chunk_name_sub)
                print("Exporting " + chunk_name + " " + chunk_name_sub + " " + str(length_sub))
                transcript.append(asr_transcript(chunk_name_sub))
                os.remove(chunk_name_sub)
            os.remove(chunk_name)
        else:
            print("Exporting " + chunk_name + " " + str(length))
            transcript.append(asr_transcript(chunk_name))
            os.remove(chunk_name)

    # chunk_length_ms = 20000  # pydub calculates in millisec
    # chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec
    # for i, chunk in enumerate(chunks):
    #     chunk_name = "{0}.wav".format(i + 1)
    #     print("exporting", chunk_name)
    #     chunk.export(chunk_name, format="wav")
    #     transcript.append(asr_transcript(chunk_name))
    #     os.remove(chunk_name)


# def STT(audio):
#     r = sr.Recognizer()
#     with audio as source:
#         audio_file = r.record(source)
#     result = r.recognize_google(audio_file)
#     return result

def STT_run():
    print(filename)

    # convert_video_to_audio_moviepy(filename)
    if (filename.endswith(".mp4") or filename.endswith(".avi")):
        convert_video_to_audio_moviepy(filename)
    audio = filename.split(".")[0] + ".wav"
    print(audio)

    splitFunction(audio)

    # transcript = asr_transcript(audio)
    INPUT = []
    global fo
    for i in transcript:
        if (len(i) > 0):
            INPUT.append(i + ". ")
    # print(INPUT)
    fo = re.sub(r'[^.\w\s]', '', "".join(INPUT))

    return fo


