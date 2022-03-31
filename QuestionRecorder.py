from scipy.io.wavfile import write

import sounddevice


def record_voice():
    fs = 44100
    second = 60
    record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=2)
    sounddevice.wait()
    sounddevice.default.device = 'digital output'
    write("Question/question.wav", fs, record_voice)

