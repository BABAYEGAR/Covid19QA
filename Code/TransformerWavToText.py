# Import necessary library

# For managing audio file
import librosa

# Importing Pytorch
import torch

# Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


def get_audio_transcription(path):
    audio, rate = librosa.load(path, sr=16000)
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Taking an input value
    input_values = tokenizer(audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    # Storing predicted ids
    prediction = torch.argmax(logits, dim=-1)
    # Passing the prediction to the tokenizer decode to get the transcription
    transcription = tokenizer.batch_decode(prediction)[0]
    return transcription


print(get_audio_transcription("Question/question.wav"))
