import pyaudio
import wave
import whisper
# from pynput import keyboard

# keyboard = Controller()
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt32, channels=1, rate=44100, input=True, frames_per_buffer=1024
)
frames = []

# Load whisper model
model = whisper.load_model("base")


def record():
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        sound_file = wave.open("sound.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt32))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()


def transcribe():
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("sound.wav")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)


if __name__ == "__main__":
    record()
    transcribe()
