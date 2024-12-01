import os
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Path to your Vosk model folder
MODEL_PATH = "/Users/gassandrid/CS/DARS/vosk-model-small-en-us-0.15"
LOG_FILE = "log.txt"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Please download a model from https://alphacephei.com/vosk/models "
              f"and unpack it as '{MODEL_PATH}' in the current folder.")
        return

    # Load the model
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, 16000)
    
    # Create a queue to hold audio data
    audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
        """Callback function to put audio data into the queue."""
        if status:
            print(f"Status: {status}", flush=True)
        audio_queue.put(bytes(indata))

    # Start the microphone input
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                           channels=1, callback=callback):
        print("Listening... Speak into the microphone.")
        previous_output = ""
        with open(LOG_FILE, "a") as log_file:
            try:
                while True:
                    # Get audio data from the queue
                    data = audio_queue.get()
                    if recognizer.AcceptWaveform(data):
                        result = recognizer.Result()
                        recognized_text = eval(result).get("text", "")
                        
                        if recognized_text:
                            # Write recognized words to the file and print them
                            log_file.write(recognized_text + "\n")
                            log_file.flush()  # Ensure the text is immediately written to the file
                            print(f"Recognized text (appended): {recognized_text}")
                            
                            # Clear the terminal and show the accumulated words
                            os.system('clear' if os.name == 'posix' else 'cls')
                            print("Recognized text:", recognized_text)
                            
                    else:
                        # Partial result or silence indicates the speaker has paused
                        if previous_output.strip():  # Check if there was prior content
                            log_file.write("\n")  # Move to the next line
                            log_file.flush()
                        previous_output = ""  # Reset output tracker
            except KeyboardInterrupt:
                print("\nExiting...")

if __name__ == "__main__":
    main()

