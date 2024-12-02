from languageModel.llm import DARSAgent
from speechRecodnition.speechRecodnition import SpeechRecognizer
from speechSynthesis.speechSynthesis import TarsVoice
import os
import keyboard
import threading
from time import sleep
import pygame
from pathlib import Path

class DARSVoiceInterface:
    def __init__(self):
        self.dars = DARSAgent()
        self.speech_recognizer = SpeechRecognizer()
        self.tars_voice = TarsVoice()
        self.is_listening = True
        self.setup_keyboard_listener()
        
    def setup_keyboard_listener(self):
        """Setup keyboard listener for spacebar toggle"""
        keyboard.on_press_key('space', self.toggle_listening)
        
    def toggle_listening(self, _=None):
        """Toggle listening state"""
        self.is_listening = not self.is_listening
        status = "activated" if self.is_listening else "deactivated"
        print(f"\nVoice input {status}")
        if not self.is_listening:
            self.tars_voice.generate_speech("Voice input deactivated. Press space to resume.")
        else:
            self.tars_voice.generate_speech("Voice input activated. I'm listening.")
        
    def run(self):
        # Initial greeting using pre-recorded audio
        greeting = "DARS initialized and ready. Press spacebar to pause or resume voice input. How can I assist you today?"
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
                if not self.is_listening:
                    sleep(0.1)  # Reduce CPU usage while paused
                    continue
                    
                # Listen for user input
                print("\nListening for your command...")
                user_input = self.speech_recognizer.listen()
                
                # Skip empty input or while paused
                if not user_input or not self.is_listening:
                    continue
                    
                print("You said:", user_input)
                
                # Check for pause/resume voice commands
                if user_input.lower() in ['pause listening', 'stop listening', 'pause input', 'stop input']:
                    self.is_listening = False
                    continue
                elif user_input.lower() in ['resume listening', 'start listening', 'resume input', 'start input']:
                    self.is_listening = True
                    continue
                
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
