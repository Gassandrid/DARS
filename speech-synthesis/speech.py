from elevenlabs import play
from elevenlabs.client import ElevenLabs
import os

client = ElevenLabs(
    api_key="YOUR_API_KEY",
)

client.text_to_speech.convert(
    voice_id="VuHE5LKSRPThk7ENDoDX",
    model_id="eleven_multilingual_v2",
    text="Hello! 你好! Hola! नमस्ते! Bonjour! こんにちは! مرحبا! 안녕하세요! Ciao! Cześć! Привіт! வணக்கம்!",
)

# class to set up the model and with a function to generat speech
class TarsVoice:
    def __init__(self):
        self.client = ElevenLabs(
                # get api key from environment variable ELEVENLABS_API_KEY
                api_key=os.getenv("ELEVENLABS_API_KEY"),
        )
        self.voice_id = "VuHE5LKSRPThk7ENDoDX"
        self.model_id = "eleven_multilingual_v2"
        print("TarsVoice initialized")
        print(f"Voice ID: {self.voice_id}")
        print(f"Model ID: {self.model_id}")
        print(f"API Key: {os.getenv('ELEVENLABS_API_KEY')}")

    def generate_speech(self, text: str):
        audio = self.client.generate(
            voice=self.voice_id,
            model=self.model_id,
            text=text,
        )
        play(audio)

def main():
    tars_voice = TarsVoice()
    generated_speech = tars_voice.generate_speech("hello my name is tars")

if __name__ == "__main__":
    main()
