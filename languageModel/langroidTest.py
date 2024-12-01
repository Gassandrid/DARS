import os
from typing import List
import fire

from langroid.pydantic_v1 import BaseModel, Field
import langroid as lr
from langroid.utils.configuration import settings
from langroid.agent.tool_message import ToolMessage
import langroid.language_models as lm
from langroid.agent.tools.orchestration import ForwardTool
from langroid.agent.chat_document import ChatDocument

# Use GPT-4o as the default LLM
DEFAULT_LLM = lm.OpenAIChatModel.GPT4o

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CityData(BaseModel):
    population: int = Field(..., description="population of city")
    country: str = Field(..., description="country of city")


class City(BaseModel):
    name: str = Field(..., description="name of city")
    details: CityData = Field(..., description="details of city")


class CityTool(lr.agent.ToolMessage):
    """Present information about a city"""

    request: str = "city_tool"
    purpose: str = """
    To present <city_info> AFTER user gives a city name,
    with all fields of the appropriate type filled out;
    DO NOT USE THIS TOOL TO ASK FOR A CITY NAME.
    SIMPLY ASK IN NATURAL LANGUAGE.
    """
    city_info: City = Field(..., description="information about a city")

    def handle(self) -> str:
        """Handle LLM's structured output if it matches City structure"""
        print("SUCCESS! Got Valid City Info")
        return """
            Thanks! Ask me for another city name; do not say anything else
            until you get a city name.
            """

    @staticmethod
    def handle_message_fallback(
        agent: lr.ChatAgent, msg: str | ChatDocument
    ) -> ForwardTool:
        """
        Forward unrecognized tool messages to the user using ForwardTool.
        """
        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            return ForwardTool(agent="User")

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                city_info=City(
                    name="San Francisco",
                    details=CityData(
                        population=800_000,
                        country="USA",
                    ),
                )
            )
        ]


def app(
    m: str = DEFAULT_LLM,  # model
    d: bool = False,  # pass -d to enable debug mode
    nc: bool = False,  # pass -nc to disable cache-retrieval
):
    settings.debug = d
    settings.cache = not nc

    # Use the API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    # Create LLM config
    llm_cfg = lm.OpenAIGPTConfig(
        api_key=api_key,  # Use the environment variable
        chat_model=m or DEFAULT_LLM,
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

        You have a humor setting, which defines how you will act, can be anything from serioos to sarcastic(with some profanity even)..
        """,
    )

    agent = lr.ChatAgent(config)
    agent.enable_message(CityTool)

    task = lr.Task(agent, interactive=False)
    task.run("Start by asking me for a city name")


if __name__ == "__main__":
    fire.Fire(app)

