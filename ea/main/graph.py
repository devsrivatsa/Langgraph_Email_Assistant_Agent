import json
from typing import TypedDict, Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, ToolMessage
from ea.main.triage import triage_input, route_after_triage
from ea.main.draft_response import draft_responses
from ea.main.find_meeting_time import find_meeting_time
from ea.main.rewrite import rewrite
from ea.main.config import get_config
from ea.main.human_inbox import (
    send_message,
    send_email_draft,
    notify,
    send_calendar_invite
)
from ea.gmail import (
    send_email,
    mark_as_read,
    send_calendar_invite
)
from ea.main.models import State

class ConfigSchema(TypedDict):
    db_id: int
    model: str


def mark_as_read_node(state: State):
    mark_as_read(state["email"]["id"])




def take_action(state: State,) -> Literal["send_message","rewrite","mark_as_read_node","find_meeting_time","send_calendar_invite","bad_tool_name"]:
    prediction = state["messages"][-1]
    if len(prediction.tool_calls) != 1:
        raise ValueError(f"Expected 1 tool call, got {len(prediction.tool_calls)} tool calls. LLM is not following the instructions.")
    tool_call = prediction.tool_calls[0]
    if tool_call["name"] == "Question":
        return "send_message"
    elif tool_call["name"] == "ResponseEmailDraft":
        return "rewrite"
    elif tool_call["name"] == "Ignore":
        return "mark_as_read_node"
    elif tool_call["name"] == "MeetingAssistant":
        return "find_meeting_time"
    elif tool_call["name"] == "SendCalendarInvite":
        return "send_calendar_invite"
    else:
        return "bad_tool_name"


def bad_tool_name(state: State):
    tool_call = state["messages"][-1].tool_calls[0]
    message = f"Could not find tool with name `{tool_call['name']}`. Make sure you are calling one of the allowed tools!"
    last_message = state["messages"][-1]
    last_message.tool_calls[0]["name"] = last_message.tool_calls[0]["name"].replace(
        ":", ""
    )
    return {
        "messages": [
            last_message,
            ToolMessage(content=message, tool_call_id=tool_call["id"]),
        ]
    }


def enter_after_human(
    state,
) -> Literal[
    "mark_as_read_node", "draft_responses", "send_email_node", "send_calendar_invite_node"
]:
    messages = state.get("messages") or []
    if len(messages) == 0:
        if state["triage"].response == "notify":
            return "mark_as_read_node"
        raise ValueError
    else:
        if isinstance(messages[-1], (ToolMessage, HumanMessage)):
            return "draft_responses"
        else:
            execute = messages[-1].tool_calls[0]
            if execute["name"] == "ResponseEmailDraft":
                return "send_email_node"
            elif execute["name"] == "SendCalendarInvite":
                return "send_calendar_invite_node"
            elif execute["name"] == "Ignore":
                return "mark_as_read_node"
            elif execute["name"] == "Question":
                return "draft_responses"
            else:
                raise ValueError


def send_calendar_invite_node(state, config):
    tool_call = state["messages"][-1].tool_calls[0]
    _args = tool_call["args"]
    email = get_config(config)["email"]
    try:
        send_calendar_invite(
            _args["emails"],
            _args["title"],
            _args["start_time"],
            _args["end_time"],
            email,
        )
        message = "Sent calendar invite!"
    except Exception as e:
        message = f"Got the following error when sending a calendar invite: {e}"
    return {"messages": [ToolMessage(content=message, tool_call_id=tool_call["id"])]}


def send_email_node(state, config):
    tool_call = state["messages"][-1].tool_calls[0]
    _args = tool_call["args"]
    email = get_config(config)["email"]
    new_receipients = _args["new_recipients"]
    if isinstance(new_receipients, str):
        new_receipients = json.loads(new_receipients)
    send_email(
        state["email"]["id"],
        _args["content"],
        email,
        addn_receipients=new_receipients,
    )
        
def human_node(state: State):
    pass

graph_builder = StateGraph(State, config_schema=ConfigSchema)

#nodes
graph_builder.add_node("human_node", human_node)
graph_builder.add_node("triage_input",triage_input)
graph_builder.add_node("draft_responses",draft_responses)
graph_builder.add_node("send_message",send_message)
graph_builder.add_node("rewrite",rewrite)
graph_builder.add_node("mark_as_read_node",mark_as_read_node)
graph_builder.add_node("send_email_draft",send_email_draft)
graph_builder.add_node("send_email_node",send_email_node)
graph_builder.add_node("bad_tool_name",bad_tool_name)
graph_builder.add_node("notify",notify)
graph_builder.add_node("send_calendar_invite_node",send_calendar_invite_node)
graph_builder.add_node("send_calendar_invite",send_calendar_invite)
graph_builder.add_node("find_meeting_time",find_meeting_time)
#entry point
graph_builder.set_entry_point("triage_input")
#edges
graph_builder.add_conditional_edges("triage_input", route_after_triage)
graph_builder.add_conditional_edges("draft_responses", take_action)
graph_builder.add_edge("send_message", "human_node")
graph_builder.add_edge("send_calendar_invite", "human_node")
graph_builder.add_edge("find_meeting_time", "draft_responses")
graph_builder.add_edge("bad_tool_name", "draft_responses")
graph_builder.add_edge("send_email_node", "mark_as_read_node") #after sending email, mark as read
graph_builder.add_edge("mark_as_read_node", END)
graph_builder.add_edge("rewrite", "send_email_draft")
graph_builder.add_edge("send_email_draft", "human_node")
graph_builder.add_edge("notify", "human_node")
graph_builder.add_conditional_edges("human_node", enter_after_human)

graph = graph_builder.compile()


# if __name__ == "__main__":
#     print("hello, world!")