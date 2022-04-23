from scipy.io.wavfile import write

import sounddevice as sd


def record_voice():
    fs = 44100
    second = 10
    sd.default.device = 'digital output'
    print("=========== Recorder Started (10 Seconds) ============")
    recorded_voice = sd.rec(int(second * fs), samplerate=fs, channels=1)
    sd.wait()
    write("Question/question.wav", fs, recorded_voice)
    print("=========== Recording Saved ============")

