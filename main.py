# import local file langroid/main.py
from languageModel.llm import DARSAgent

# initialize DARSAgent
dars = DARSAgent()

# run the agent
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    natural_language, function_output = dars.process_message(user_input)
    if function_output:
        print("Function:", function_output)
    if natural_language:
        print("DARS says:", natural_language)
