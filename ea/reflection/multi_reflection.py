from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START
from langgraph.types import Command, Send
from ea.reflection.utils import get_trajectory_clean
from ea.reflection.general_reflection import call_reflection

TONE_INSTRUCTIONS = """Only update the prompt to include instructions on the **style and tone and format** of the response. 
Do NOT update the prompt to include anything about the actual content - only the style and tone and format. 
The user sometimes responds differently to different types of people - take that into account but don't be too specific.
"""

RESPONSE_INSTRUCTIONS = """Only update the prompt to include instructions on the **content** of the response. 
Do NOT update the prompt to include anything about the tone or style or format of the response.
"""

SCHEDULE_INSTRUCTIONS = """Only update the prompt to include instructions on how to send calendar invites - example when to send them, the the title of the invite should be, length, time of day, etc."""

BACKGROUND_INSTRUCTIONS = """Only update the prompt to include pieces of information that are relevant to being the user's assistant. 
Do not update the instructions to include anything about the tone of emails sent, when to send calendar invites.
Examples of good things to include are are (but not limited to): people's emails, addresses, etc."""


CHOOSE_MEMORY_PROMPT = """You are an ai agent who improves prompt by changing them based on 1. agent's trajectory, 2. user's feedback, and 3. various options given to you.

Here is the agent's trajectory:
<trajectory>
{trajectory}
</trajectory>

Here is the user's feedback:
<feedback>
{feedback}
</feedback>

Here are the options - types of prompts that you can update in order to change their behavior:
<types_of_prompts>
{types_of_prompts}
</types_of_prompts>

Please choose the types of prompts that are worth updating based on the trajectory, and feedback. Only do this if the feedback seems like it has information relevant to the prompt.
You will update the prompts themselves in a separate step.
You do not have to update any memory types if you don't want to! Just leave them empty.
"""


MEMORY_TO_UPDATE = {
    "tone": "Instructions on the tone of the email. Update this if you learn new information about the tone in which the user likes to respond that may be relevant in the future emails.",

    "background": "Background information about the user. Update this if you learn new information about the user that may be relevant in the future.",
    
    "email": """Instructions about the type of content to be included in the email. Update this if you learn new information about how the user likes to respond to email 
    (not the tone, and not the information about the user, but specifically about how or when they like to respond to email) that may be relevant in the future.""",
    
    "calendar":"Instructions on how to update calendar invites (including title, length, time, etc). Update this if you learn new information about how the user likes to schedule events that may be relevant in future emails."
}


MEMORY_TO_UPDATE_INSTRUCTIONS = {
    "tone": TONE_INSTRUCTIONS,
    "background": BACKGROUND_INSTRUCTIONS,
    "email": RESPONSE_INSTRUCTIONS,
    "calendar": SCHEDULE_INSTRUCTIONS
}



class MultiMemoryInput(TypedDict):
    prompt_types: list[str]
    feedback: str
    assistant_key: str


async def determine_what_to_update(state: MultiMemoryInput):
    reflection_model = ChatOpenAI(model="gpt-4o", disable_streaming=True)
    trajectory = get_trajectory_clean(state["messages"])
    types_of_prompts = "\n".join([f"`{p_type}`: {MEMORY_TO_UPDATE[p_type]}" for p_type in state["prompt_types"]])
    prompt = CHOOSE_MEMORY_PROMPT.format(
        trajectory=trajectory,
        feedback=state["feedback"],
        types_of_prompts=types_of_prompts
    )
    
    class MemoryToUpdate(TypedDict):
        memory_types_to_update: list[str]
    
    response = reflection_model.with_structured_output(MemoryToUpdate).invoke(prompt)
    sends = []
    for mem_type in response.memory_types_to_update:
        _state = {
            "messages": state["messages"],
            "feedback": state["feedback"],
            "prompt_key": MEMORY_TO_UPDATE[mem_type],
            "assistant_key": state["assistant_key"],
            "instructions": MEMORY_TO_UPDATE_INSTRUCTIONS[mem_type]
        }
        send = Send("reflection", _state)
        sends.append(send)
    
    return Command(goto=sends)



multi_reflection_graph = StateGraph(MultiMemoryInput)

multi_reflection_graph.add_node(determine_what_to_update)
multi_reflection_graph.add_node("reflection", call_reflection)
multi_reflection_graph.add_edge(START, "determine_what_to_update")

multi_reflection_graph = multi_reflection_graph.compile()