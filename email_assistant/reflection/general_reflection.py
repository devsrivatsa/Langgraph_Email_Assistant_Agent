from utils import get_trajectory_clean
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI
from typing import Optional
from typing_extensions import TypedDict

class ReflectionState(MessagesState):
    feedback: Optional[str]
    prompt_key: str
    assistant_key: str
    instructions: str


class GeneralResponse(TypedDict):
    logic: str
    update_prompt: bool
    new_prompt: str


general_reflection_prompt = """Your task is to help an ai agent improve by only changing their system prompt based on the user's feedback and the agent's trajectory.

This is the current prompt:
<current_prompt>
{current_prompt}
</current_prompt>

This is the agent's trajectory:
<trajectory>
{trajectory}
</trajectory>

This is the user's feedback:
<feedback>
{feedback}
</feedback>

Here are the instructions for updating the agent's prompt:
<instructions>
{instructions}
</instructions>

Based on this, return an updated prompt.

Note: 
You should return the full prompt, so if there's anything from before that you want to include, make sure to do that. 
Feel free to override or change anything that seem irrelevant. 
You do not need to update the prompt if you deem it unnecessary, just return `update_prompt = False` and an empty string for new prompt.
"""

async def get_output(reflection_model, messages, current_prompt, feedback, instructions):
    trajectory = get_trajectory_clean(messages)
    prompt = general_reflection_prompt.format(
        current_prompt=current_prompt,
        trajectory=trajectory,
        feedback=feedback,
        instructions=instructions
    )
    output = await reflection_model.with_structured_output(GeneralResponse, method="json_schema").ainvoke(prompt)
    return output

async def update_general(state: ReflectionState, config, store: BaseStore):
    reflection_model =ChatOpenAI(model="o1", disable_streaming=True)
    namespace = (state["assistant_key"],)
    key = state["prompt_key"]
    result = await store.aget(namespace, key)

    output = await get_output(
        reflection_model,
        state["messages"],
        result.value["data"],
        state["feedback"],
        state["instructions"]
    )
    if output["update_prompt"]:
        await store.aput(namespace, key, {"data": output["new_prompt"]}, index=False)


general_reflection_graph = StateGraph(ReflectionState)
general_reflection_graph.add_node(update_general)
general_reflection_graph.add_edge(START, "update_general")
general_reflection_graph.add_edge("update_general", END)
general_reflection_graph = general_reflection_graph.compile()


async def call_reflection(state: ReflectionState):
    await general_reflection_graph.ainvoke(state)