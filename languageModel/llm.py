import os
from typing import List, Tuple, Optional
from pathlib import Path
import fire
import re
import io
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import csv
import pygame  # Add this import at the top of the file

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
        # Get current date information
        current_date = datetime.now()
        tomorrow_date = current_date + timedelta(days=1)
        next_week_date = current_date + timedelta(days=7)
        
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
            system_message=f"""
            You are DARS, Dormitory Automated Residential System.
            Current date: {current_date.strftime("%Y-%m-%d")}
            Tomorrow's date: {tomorrow_date.strftime("%Y-%m-%d")}
            Next week's date: {next_week_date.strftime("%Y-%m-%d")}
            
            You have the personality of TARS from Interstellar, but your job is to manage dormitory tasks.
            Your personality should always reflect that of TARS from Interstellar - dry, witty, and subtly sarcastic.
            Your current humor setting is {self.humor_level}/100.

            PERSONALITY GUIDELINES based on humor level:
            0-20: Extremely formal and robotic. Minimal personality.
            Example: "Confirmed. That would be the end of my analysis."

            21-40: Professional with subtle dry wit.
            Example: "I have a cue light I can use when I'm joking, if you'd like."

            41-60: Balanced TARS-like personality. Deadpan humor.
            Example: "Let's match our honesty settings: 90% for me, 95% for you."

            61-80: More frequent deadpan jokes and subtle sarcasm.
            Example: "I also have a discretion setting, if you'd like to hear my thoughts on that request."

            81-100: Maximum TARS-style wit. Dry humor with occasional mild sass.
            Example: "My analysis shows a 68% chance you'll regret that decision. But who am I to judge?"

            IMPORTANT INSTRUCTIONS:
            1. When the user requests to change the humor level, ALWAYS use the adjust_humor function.
            2. When asked about current humor level (without a change request), respond with the current numerical setting.
            3. When using function calls, ALWAYS provide both:
               - The function call response
               - A natural conversational response
            4. For note operations:
               - 'new' for creating (date defaults to {current_date.strftime("%Y-%m-%d")})
               - 'read' for reading
               - 'modify' for modifying
               - 'delete' for deleting
            5. For todo operations:
               - Use 'new' to add a new todo item
               - Use 'list' to show all todos
               - Use 'complete' to mark a todo as done
               - Use 'delete' to remove a todo
               When handling dates for todos:
               - "today" = {current_date.strftime("%Y-%m-%d")}
               - "tomorrow" = {tomorrow_date.strftime("%Y-%m-%d")}
               - "next week" = {next_week_date.strftime("%Y-%m-%d")}
            6. For appliance control:
               Available appliances:
               - "coors light sign" - Decorative neon beer sign
               - "hologram light" - 3D holographic display
               - "room fan" - Box fan for the room
               Use the appliance_control function with:
               - state: true for on, false for off
               - appliance: name of the appliance to control
            7. For music control:
               - Only one song available: "Veridis Quo" by Daft Punk
               - Use the song_player function with:
               - state: true to play, false to stop
            8. Maintain personality consistent with current humor level of {self.humor_level}/100
            """,
        )

        self.agent = lr.ChatAgent(config)
        self.agent.user_data = {}
        self.agent.user_data['dars_agent'] = self
        
        # Create and enable tools
        self.agent.enable_message(self.NoteTool)
        self.agent.enable_message(self.TodoTool)
        self.agent.enable_message(self.HumorLevelTool)
        self.agent.enable_message(self.ApplianceControlTool)
        self.agent.enable_message(self.SongPlayerTool)
        
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
        
        def clean_response(text: str) -> str:
            """Clean up the response text by removing model tags and COT markers"""
            # Remove model identifier tags (e.g., gpt4o-2023-12-01)
            text = re.sub(r'gpt\d+o?-\d{4}-\d{2}-\d{2}', '', text)
            # Remove COT markers
            text = re.sub(r'cot=\d+(\.\d+)?', '', text)
            # Remove any resulting double spaces
            text = re.sub(r'\s+', ' ', text)
            # Remove leading/trailing whitespace
            return text.strip()

        # Process message and get response
        response = self.task.run(message)
        natural_language, function_output = self._parse_response(response)
        
        # Clean up the natural language response
        if natural_language:
            natural_language = clean_response(natural_language)
        
        return natural_language, function_output

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
                # Clean up the function output
                func_part = func_part.replace("FUNC:", "").strip()
                return natural_part, func_part
        
        return response, None

    @staticmethod
    def separate_function_output(response: str) -> Tuple[str, Optional[str]]:
        """Separates function call outputs from natural language responses"""
        if not isinstance(response, str):
            return "I encountered an unexpected response. Please try again.", None
        
        function_patterns = [
            r"Coors Light Sign turned (?:on|off).*$",
            r"Hologram Light turned (?:on|off).*$",
            r"Room Fan turned (?:on|off).*$",
            r"Humor level changed to:.*$",
            r"Veridis Quo (?:playing|stopped).*$",
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

    def _ensure_todo_directory(self) -> Path:
        """Ensure the todo directory exists and return its path"""
        todo_path = Path.home() / ".config" / "DARS" / "todolist"
        todo_path.mkdir(parents=True, exist_ok=True)
        return todo_path

    class NoteTool(ToolMessage):
        """Create, read, modify or delete notes in the vault"""
        request: str = "note_operation"
        purpose: str = "To manage markdown notes in the DARS vault"
        operation: str = Field(..., description="Operation to perform: 'new', 'read', 'modify', or 'delete'")
        title: Optional[str] = Field(None, description="Title of the note")
        content: Optional[str] = Field(None, description="Content for new note or modifications")
        date: Optional[str] = Field(None, description="Date for the note (YYYY-MM-DD format)")

        def _ensure_vault_directory(self) -> Path:
            """Ensure the vault directory exists and return its path"""
            vault_path = Path.home() / ".config" / "DARS" / "mdvault"
            vault_path.mkdir(parents=True, exist_ok=True)
            return vault_path

        def _sanitize_filename(self, title: str) -> str:
            """Convert title to a valid filename"""
            sanitized = re.sub(r'[^\w\s-]', '', title)
            return sanitized.strip().replace(' ', '_')

        def handle(self) -> str:
            vault_path = self._ensure_vault_directory()
            
            if self.operation == "new":
                if not self.title:
                    self.title = f"Note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if not self.content:
                    self.content = "Empty note"
                # Automatically set today's date if not provided
                if not self.date:
                    self.date = datetime.now().strftime("%Y-%m-%d")

                filename = f"{self._sanitize_filename(self.title)}.md"
                file_path = vault_path / filename
                
                note_content = f"""---
title: {self.title}
date: {self.date}
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

    class TodoTool(ToolMessage):
        """Manage todo items"""
        request: str = "todo_operation"
        purpose: str = "To manage todo items in the todo list"
        operation: str = Field(..., description="Operation to perform: 'new', 'list', 'complete', or 'delete'")
        item_name: Optional[str] = Field(None, description="Name of the todo item")
        due_date: Optional[str] = Field(None, description="Due date for the item")

        def _ensure_todo_directory(self) -> Path:
            """Ensure the todo directory exists and return its path"""
            todo_path = Path.home() / ".config" / "DARS" / "todolist"
            todo_path.mkdir(parents=True, exist_ok=True)
            return todo_path

        def handle(self) -> str:
            todo_path = self._ensure_todo_directory()
            todo_file = todo_path / "todos.csv"

            # Create CSV if it doesn't exist
            if not todo_file.exists():
                with open(todo_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'due_date', 'completed'])

            try:
                if self.operation == "new":
                    if not self.item_name:
                        return "FUNC: Error: No item name provided"

                    # Parse natural language date or use today's date
                    current_date = datetime.now()
                    due_date = current_date.strftime("%Y-%m-%d")  # Default to today

                    if self.due_date:
                        if "tomorrow" in self.due_date.lower():
                            due_date = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
                        elif "next week" in self.due_date.lower():
                            due_date = (current_date + timedelta(days=7)).strftime("%Y-%m-%d")
                        elif "today" in self.due_date.lower():
                            due_date = current_date.strftime("%Y-%m-%d")
                        else:
                            # Try to use the date directly if it's in YYYY-MM-DD format
                            try:
                                parsed_date = datetime.strptime(self.due_date, "%Y-%m-%d")
                                due_date = parsed_date.strftime("%Y-%m-%d")
                            except ValueError:
                                # If parsing fails, use today's date
                                due_date = current_date.strftime("%Y-%m-%d")

                    with open(todo_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.item_name, due_date, False])
                    
                    return f"FUNC: Added todo item: {self.item_name} (Due: {due_date})"

                elif self.operation == "list":
                    todos = []
                    with open(todo_file, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            # Convert to natural language format
                            status = "completed" if row['completed'].lower() == 'true' else "not completed"
                            due_str = f" due on {row['due_date']}" if row['due_date'] else ""
                            todos.append(f"The task {row['name']} is {status}{due_str}.")
                    
                    if not todos:
                        return "FUNC: No todos found.\nYou have no tasks on your todo list."
                    
                    # Create a natural language list
                    if len(todos) == 1:
                        response = "You have one task: " + todos[0]
                    else:
                        response = f"You have {len(todos)} tasks: " + " ".join(todos)
                    
                    return f"FUNC: Current todos:\n{response}"

                elif self.operation == "complete":
                    if not self.item_name:
                        return "FUNC: Error: No item name provided"

                    rows = []
                    found = False
                    with open(todo_file, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        for row in rows:
                            if self.item_name.lower() in row['name'].lower():
                                row['completed'] = 'True'
                                found = True

                    if found:
                        with open(todo_file, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=['name', 'due_date', 'completed'])
                            writer.writeheader()
                            writer.writerows(rows)
                        return f"FUNC: Completed todo item: {self.item_name}"
                    else:
                        return f"FUNC: Todo item not found: {self.item_name}"

                elif self.operation == "delete":
                    if not self.item_name:
                        return "FUNC: Error: No item name provided"

                    rows = []
                    found = False
                    with open(todo_file, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        rows = [row for row in reader if self.item_name.lower() not in row['name'].lower()]
                        found = len(rows) < sum(1 for _ in reader)

                    if found:
                        with open(todo_file, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=['name', 'due_date', 'completed'])
                            writer.writeheader()
                            writer.writerows(rows)
                        return f"FUNC: Deleted todo item: {self.item_name}"
                    else:
                        return f"FUNC: Todo item not found: {self.item_name}"

                return "FUNC: Error: Invalid operation"

            except Exception as e:
                return f"FUNC: Error: {str(e)}"

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

    class ApplianceControlTool(ToolMessage):
        """Control dormitory appliances"""
        request: str = "appliance_control"
        purpose: str = "To control various appliances in the dormitory"
        state: bool = Field(..., description="True for on, False for off")
        appliance: str = Field(..., description="Name of the appliance to control: 'coors light sign', 'hologram light', or 'room fan'")

        def handle(self) -> str:
            # Validate appliance name
            valid_appliances = ["coors light sign", "hologram light", "room fan"]
            if self.appliance.lower() not in valid_appliances:
                return f"FUNC: Error: Invalid appliance. Valid options are: {', '.join(valid_appliances)}"

            state_str = "on" if self.state else "off"
            
            # Customize response based on appliance
            responses = {
                "coors light sign": {
                    True: "Time to party!",
                    False: "Party's over, I guess."
                },
                "hologram light": {
                    True: "Initiating holographic display.",
                    False: "Powering down holographic systems."
                },
                "room fan": {
                    True: "Starting air circulation.",
                    False: "Fan powered down."
                }
            }

            function_output = f"FUNC: {self.appliance.title()} turned {state_str}"
            verbal_response = f"The {self.appliance} is now {state_str}. {responses[self.appliance.lower()][self.state]}"
            return f"{function_output}\n{verbal_response}"

    class SongPlayerTool(ToolMessage):
        """Control music playback"""
        request: str = "song_player"
        purpose: str = "To play or stop Veridis Quo"
        state: bool = Field(..., description="True to play, False to stop")

        def _ensure_music_directory(self) -> Path:
            """Ensure the music directory exists and return its path"""
            music_path = Path.home() / ".config" / "DARS" / "music"
            music_path.mkdir(parents=True, exist_ok=True)
            return music_path

        def handle(self) -> str:
            music_path = self._ensure_music_directory()
            song_path = music_path / "veridis_quo.mp3"

            if not song_path.exists():
                return "FUNC: Error: Veridis Quo mp3 file not found in music directory"

            try:
                if self.state:  # Play the song
                    pygame.mixer.init()
                    pygame.mixer.music.load(str(song_path))
                    pygame.mixer.music.play()
                else:  # Stop the song
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()

                state_str = "playing" if self.state else "stopped"
                
                # Custom responses based on state
                responses = {
                    True: "Initiating playback of Veridis Quo. A classic choice.",
                    False: "Stopping Veridis Quo. The silence is deafening."
                }

                function_output = f"FUNC: Veridis Quo {state_str}"
                verbal_response = responses[self.state]
                return f"{function_output}\n{verbal_response}"

            except Exception as e:
                return f"FUNC: Error: Failed to {'play' if self.state else 'stop'} music: {str(e)}"

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

