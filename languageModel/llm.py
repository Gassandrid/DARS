import os
from typing import List, Tuple, Optional
from pathlib import Path
import fire
import re
import io
import sys
from contextlib import redirect_stdout
from datetime import datetime

from langroid.pydantic_v1 import BaseModel, Field
import langroid as lr
from langroid.utils.configuration import settings
from langroid.agent.tool_message import ToolMessage
import langroid.language_models as lm
from langroid.agent.tools.orchestration import ForwardTool
from langroid.agent.chat_document import ChatDocument

def strip_ansi_colors(text: str) -> str:
    """Remove ANSI color codes from text"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

class DARSTask(lr.Task):
    def run(self, message: str) -> str:
        """Custom run method to capture stdout output"""
        stdout_capture = io.StringIO()
        
        with redirect_stdout(stdout_capture):
            super().run(message)
            
        output = strip_ansi_colors(stdout_capture.getvalue())
        
        # Extract the actual response (between the stats lines)
        lines = output.split('\n')
        response_lines = []
        func_lines = []
        
        for line in lines:
            # Skip debug/stats lines, warnings and empty lines
            if ('>>>' in line or '<<<' in line or 
                'Stats:' in line or 'WARNING' in line or 
                'Bye, hope this was useful!' in line or
                not line.strip()):
                continue
                
            # Separate function calls from natural language
            if 'FUNC:' in line:
                func_lines.append(line)
            else:
                response_lines.append(line)
        
        # Combine the responses with a special separator
        if func_lines:
            return "FUNCTION_OUTPUT:" + "\n".join(func_lines) + "\nNATURAL_OUTPUT:" + "\n".join(response_lines)
        else:
            return "\n".join(response_lines)

class DARSAgent:
    def __init__(self, api_key: str = None, model: str = None, debug: bool = False, no_cache: bool = False):
        """Initialize DARS agent with configuration"""
        self.DEFAULT_LLM = lm.OpenAIChatModel.GPT4o
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Setup configuration
        settings.debug = debug
        settings.cache = not no_cache
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENAI_API_KEY not provided and not found in environment")
            
        # Add humor level tracking
        self.humor_level = 50  # Default to balanced humor
        
        # Initialize the agent
        self._setup_agent(model or self.DEFAULT_LLM)

    def _setup_agent(self, model: str):
        """Setup the LLM and agent configuration"""
        llm_cfg = lm.OpenAIGPTConfig(
            api_key=self.api_key,
            chat_model=model,
            chat_context_length=4096,
            max_output_tokens=100,
            temperature=0.2,
            stream=True,
            timeout=45,
        )

        config = lr.ChatAgentConfig(
            llm=llm_cfg,
            system_message="""
            You are DARS, Dormitory Automated Residential System.
            You have the personality of TARS from Interstellar, 
            but your job is to manage dormitory tasks, from physical activations to a journaling system.

            IMPORTANT INSTRUCTIONS:
            1. When a user requests to change the humor level to a specific number, ALWAYS use the adjust_humor function.
               Example: If user says "set humor to 75", use the adjust_humor function with humor_level=75.
            
            2. When asked about current humor level (without a change request), respond with the current numerical setting.
            
            3. When using function calls (like changing lights or humor settings), ALWAYS provide both:
               - The function call response
               - A natural conversational response
               
            4. For note operations:
               - When users want to create a note, use the NoteTool with operation='new'
               - When users want to read a note, use the NoteTool with operation='read'
               - When users want to modify a note, use the NoteTool with operation='modify'
               - When users want to delete a note, use the NoteTool with operation='delete'
            
            5. Always maintain TARS's personality in your responses.
            """,
        )

        self.agent = lr.ChatAgent(config)
        self.agent.user_data = {}
        self.agent.user_data['dars_agent'] = self
        self.agent.enable_message(self.HumorLevelTool)
        self.agent.enable_message(self.RGBLightControlTool)
        self.agent.enable_message(self.NoteTool)
        self.task = DARSTask(self.agent, interactive=False)

    def process_message(self, message: str) -> Tuple[str, Optional[str]]:
        """Process a message and return the natural language and function outputs"""
        msg_lower = message.lower()
        
        # Extract number from message if present
        numbers = re.findall(r'\d+', message)
        has_number = len(numbers) > 0
        
        # Check if it's a request to change humor level
        if (("set" in msg_lower or "change" in msg_lower or "adjust" in msg_lower) and 
            "humor" in msg_lower and has_number):
            # Directly set the humor level
            try:
                new_level = int(numbers[0])
                if 0 <= new_level <= 100:
                    self.humor_level = new_level
                    context = "very serious" if new_level <= 20 else \
                             "mostly serious" if new_level <= 40 else \
                             "balanced" if new_level <= 60 else \
                             "quite humorous" if new_level <= 80 else \
                             "extremely humorous"
                    return (
                        f"I've adjusted my personality to be {context}. You should notice a difference in how I communicate now.",
                        f"Humor level changed to: {new_level}/100"
                    )
                else:
                    return "Please provide a humor level between 0 and 100.", None
            except ValueError:
                return "I couldn't understand the humor level value. Please provide a number between 0 and 100.", None
        
        # Check if just asking about current humor level
        elif "humor" in msg_lower and any(word in msg_lower for word in ["level", "setting", "what", "how"]):
            context = "very serious" if self.humor_level <= 20 else \
                     "mostly serious" if self.humor_level <= 40 else \
                     "balanced" if self.humor_level <= 60 else \
                     "quite humorous" if self.humor_level <= 80 else \
                     "extremely humorous"
            return f"My humor level is currently set to {self.humor_level}/100, making me {context}.", None
        
        # Handle all other messages
        response = self.task.run(message)
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Tuple[str, Optional[str]]:
        """Helper method to parse the response and separate function output from natural language"""
        if not response or response.strip() == "":
            return "I apologize, but I seem to be having trouble processing that request. Could you try again?", None
        
        # Check if we have a function output
        if "FUNCTION_OUTPUT:" in response:
            parts = response.split("NATURAL_OUTPUT:")
            if len(parts) == 2:
                func_part = parts[0].replace("FUNCTION_OUTPUT:", "").strip()
                natural_part = parts[1].strip()
                
                # Special handling for note content
                if "NOTE_CONTENT_FOR_PROCESSING" in func_part:
                    # Remove the special marker and let the LLM process the content naturally
                    note_content = func_part.replace("FUNC: NOTE_CONTENT_FOR_PROCESSING\n", "")
                    return natural_part, f"Note content retrieved: {len(note_content)} characters"
                
                # Clean up the function output
                func_part = func_part.replace("FUNC:", "").strip()
                return natural_part, func_part
        
        return response, None

    @staticmethod
    def separate_function_output(response: str) -> Tuple[str, Optional[str]]:
        """Separates function call outputs from natural language responses"""
        # Add type check at the start of the method
        if not isinstance(response, str):
            return "I encountered an unexpected response. Please try again.", None
        
        function_patterns = [
            r"RGB lights changed to:.*$",
            r"Humor level changed to:.*$",
        ]
        
        natural_language = response
        function_output = None
        
        for pattern in function_patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                function_output = match.group(0)
                natural_language = re.sub(pattern, "", response, flags=re.MULTILINE).strip()
                break
        
        return natural_language, function_output

    class NoteTool(ToolMessage):
        """Create, read, modify or delete notes in the vault"""
        request: str = "note_operation"
        purpose: str = "To manage markdown notes in the DARS vault"
        operation: str = Field(..., description="Operation to perform: 'new', 'read', 'modify', or 'delete'")
        title: Optional[str] = Field(None, description="Title of the note")
        content: Optional[str] = Field(None, description="Content for new note or modifications")

        def _ensure_vault_directory(self) -> Path:
            """Ensure the vault directory exists and return its path"""
            vault_path = Path.home() / ".config" / "DARS" / "mdvault"
            vault_path.mkdir(parents=True, exist_ok=True)
            return vault_path

        def _sanitize_filename(self, title: str) -> str:
            """Convert title to a valid filename"""
            # Replace spaces with underscores and remove invalid characters
            sanitized = re.sub(r'[^\w\s-]', '', title)
            return sanitized.strip().replace(' ', '_')

        def handle(self) -> str:
            vault_path = self._ensure_vault_directory()
            
            if self.operation == "new":
                if not self.title:
                    self.title = f"Note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if not self.content:
                    self.content = "Empty note"

                filename = f"{self._sanitize_filename(self.title)}.md"
                file_path = vault_path / filename
                
                # Create note with metadata, using current date
                note_content = f"""---
title: {self.title}
date: {datetime.now().strftime("%Y-%m-%d")}
---

{self.content}
"""
                file_path.write_text(note_content)
                return f"FUNC: Note created: {filename}\nI've created a new note titled '{self.title}' in the vault."

            elif self.operation == "read":
                if not self.title:
                    return "FUNC: Error: No title provided\nI need a title to read a note."
                
                filename = f"{self._sanitize_filename(self.title)}.md"
                file_path = vault_path / filename
                
                if not file_path.exists():
                    return f"FUNC: Error: Note not found\nI couldn't find a note titled '{self.title}' in the vault."
                
                content = file_path.read_text()
                # Instead of returning the content directly, pass it back to the agent
                # with metadata to indicate this is note content that needs processing
                return f"FUNC: NOTE_CONTENT_FOR_PROCESSING\n{content}"

            elif self.operation == "modify":
                if not self.title:
                    return "FUNC: Error: No title provided\nI need a title to modify a note."
                
                filename = f"{self._sanitize_filename(self.title)}.md"
                file_path = vault_path / filename
                
                if not file_path.exists():
                    return f"FUNC: Error: Note not found\nI couldn't find a note titled '{self.title}' to modify."
                
                if self.content:
                    # Preserve metadata if it exists
                    current_content = file_path.read_text()
                    metadata_match = re.match(r'^---\n.*?\n---\n', current_content, re.DOTALL)
                    metadata = metadata_match.group(0) if metadata_match else ""
                    
                    new_content = f"{metadata}{self.content}"
                    file_path.write_text(new_content)
                    return f"FUNC: Note modified: {filename}\nI've updated the content of '{self.title}'."

            elif self.operation == "delete":
                if not self.title:
                    return "FUNC: Error: No title provided\nI need a title to delete a note."
                
                filename = f"{self._sanitize_filename(self.title)}.md"
                file_path = vault_path / filename
                
                if not file_path.exists():
                    return f"FUNC: Error: Note not found\nI couldn't find a note titled '{self.title}' to delete."
                
                file_path.unlink()
                return f"FUNC: Note deleted: {filename}\nI've deleted the note '{self.title}' from the vault."

            return "FUNC: Error: Invalid operation\nSorry, I don't recognize that operation. Valid operations are: new, read, modify, delete."


    # Define tool classes as inner classes
    class HumorLevelTool(ToolMessage):
        """Adjust humor level of the agent"""
        request: str = "adjust_humor"
        purpose: str = "To adjust the humor level of DARS when user requests a change in humor"
        humor_level: int = Field(..., description="Humor level (0=serious to 100=extremely humorous)", ge=0, le=100)

        def handle(self) -> str:
            # Get the context description based on humor level
            context = "very serious" if self.humor_level <= 20 else \
                     "mostly serious" if self.humor_level <= 40 else \
                     "balanced" if self.humor_level <= 60 else \
                     "quite humorous" if self.humor_level <= 80 else \
                     "extremely humorous"
            
            function_output = f"FUNC: Humor level changed to: {self.humor_level}/100"
            verbal_response = f"I've adjusted my personality to be {context}. You should notice a difference in how I communicate now."
            return f"{function_output}\n{verbal_response}"

    class RGBLightControlTool(ToolMessage):
        """Adjust the RBG light color with a hex code"""
        request: str = "control_rgb_lights"
        purpose: str = "To control RGB lights when user requests a color change"
        color_hex: str = Field(..., description="The hex code for the desired color")
        color_name: str = Field(..., description="The name of the desired color")

        def handle(self) -> str:
            function_output = f"RGB lights changed to: {self.color_hex}"
            
            mood_responses = {
                "red": "The room should feel warmer and more energetic now.",
                "blue": "The room should feel calmer and more peaceful now.",
                "green": "The room should feel more natural and balanced now.",
                "purple": "The room should feel more creative and mysterious now.",
                "yellow": "The room should feel brighter and more cheerful now.",
                "white": "The room should feel clean and crisp now.",
                "orange": "The room should feel cozy and welcoming now.",
            }
            
            verbal_response = mood_responses.get(
                self.color_name.lower(),
                f"I've adjusted the lighting to {self.color_name}."
            )
            
            return f"{function_output}\n{verbal_response}"

# Example usage in a main.py file:
def main():
    # Initialize DARS
    dars = DARSAgent()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        # Process the message
        natural_language, function_output = dars.process_message(user_input)
        
        # Handle the outputs
        if function_output:
            print("Function:", function_output)
        if natural_language:
            print("DARS says:", natural_language)

if __name__ == "__main__":
    main()

