from languageModel.llm import DARSAgent
from speechRecognition.speechRecognition import SpeechRecognizer
from speechSynthesis.speechSynthesis import TarsVoice
import os
import pygame
from pathlib import Path

class DARSVoiceInterface:
    def __init__(self):
        self.dars = DARSAgent()
        self.speech_recognizer = SpeechRecognizer()
        self.tars_voice = TarsVoice()
        
    def run(self):
        # Initial greeting using pre-recorded audio
        greeting = "DARS initialized and ready. Press enter to start voice input. How can I assist you today?"
        print("DARS says:", greeting)
        
        # Play pre-recorded greeting
        pygame.mixer.init()
        pygame.mixer.music.load(str(Path.home() / ".config" / "dars" / "dars_greeting.mp3"))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait for the greeting to finish
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
        
        while True:
            try:
                # Wait for Enter key
                input("\nPress Enter to start listening...")
                
                # Listen for user input
                print("Listening for your command...")
                user_input = self.speech_recognizer.listen()
                print("You said:", user_input)
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'stop', 'goodbye']:
                    farewell = "Shutting down DARS. Goodbye!"
                    print("DARS says:", farewell)
                    self.tars_voice.generate_speech(farewell)
                    break
                
                # Process the input through DARS
                natural_language, function_output = self.dars.process_message(user_input)
                
                # Handle function output if present
                if function_output:
                    print("Function:", function_output)
                
                # Convert DARS's response to speech
                if natural_language:
                    print("DARS says:", natural_language)
                    self.tars_voice.generate_speech(natural_language)
                
            except KeyboardInterrupt:
                print("\nInterrupt received, shutting down...")
                break
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                print(error_msg)
                self.tars_voice.generate_speech("I encountered an error. Please try again.")

def main():
    # Check for required environment variables
    if not os.getenv("ELEVENLABS_API_KEY"):
        print("Error: ELEVENLABS_API_KEY environment variable not set")
        return
        
    try:
        dars_interface = DARSVoiceInterface()
        dars_interface.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()
