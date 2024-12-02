import os
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from pathlib import Path
from typing import Optional

class SpeechRecognizer:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the speech recognizer with optional custom model path"""
        self.MODEL_PATH = model_path or "/users/ewan/DARS/vosk-model-small-en-us-0.15"
        self.audio_queue = queue.Queue()
        self._setup_model()

    def _setup_model(self) -> None:
        """Set up the Vosk model and recognizer"""
        if not os.path.exists(self.MODEL_PATH):
            raise FileNotFoundError(
                f"Please download a model from https://alphacephei.com/vosk/models "
                f"and unpack it as '{self.MODEL_PATH}' in the current folder."
            )
        self.model = Model(self.MODEL_PATH)
        self.recognizer = KaldiRecognizer(self.model, 16000)

    def _audio_callback(self, indata, frames, time, status) -> None:
        """Callback function to put audio data into the queue."""
        if status:
            print(f"Status: {status}", flush=True)
        self.audio_queue.put(bytes(indata))

    def listen(self) -> str:
        """Listen for a complete sentence and return the recognized text.
        Returns when the user stops speaking."""
        
        recognized_text = ""
        silence_counter = 0  # Count frames of silence
        
        # Start the microphone input
        with sd.RawInputStream(
            samplerate=16000,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self._audio_callback
        ):
            print("Listening... Speak into the microphone.")
            
            try:
                while True:
                    # Get audio data from the queue
                    data = self.audio_queue.get()
                    
                    if self.recognizer.AcceptWaveform(data):
                        result = eval(self.recognizer.Result())
                        text = result.get("text", "").strip()
                        
                        if text:
                            recognized_text = text
                            silence_counter = 0  # Reset silence counter
                        else:
                            silence_counter += 1
                            
                        # If we have text and detect silence, return the result
                        if recognized_text and silence_counter >= 2:
                            return recognized_text
                            
                    # Partial results indicate ongoing speech
                    else:
                        partial = eval(self.recognizer.PartialResult())
                        if not partial.get("partial", "").strip():
                            silence_counter += 1
                        else:
                            silence_counter = 0
                            
                        # Return recognized text after sustained silence
                        if recognized_text and silence_counter >= 3:
                            return recognized_text
                            
            except KeyboardInterrupt:
                return recognized_text if recognized_text else "Recognition interrupted"
            except Exception as e:
                return f"Error during recognition: {str(e)}"

    def new(self) -> 'SpeechRecognizer':
        """Create and return a new instance of the speech recognizer"""
        return SpeechRecognizer(self.MODEL_PATH)

# Example usage:
if __name__ == "__main__":
    try:
        recognizer = SpeechRecognizer()
        while True:
            text = recognizer.listen()
            if text:
                print(f"Recognized: {text}")
            
            # Optional: break condition
            if text.lower() in ['quit', 'exit', 'stop']:
                break
                
    except KeyboardInterrupt:
        print("\nStopping speech recognition...")

