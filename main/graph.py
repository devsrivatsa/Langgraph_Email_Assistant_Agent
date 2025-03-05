import json
from typing import TypedDict, Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, ToolMessage
from triage import triage_input
from draft_response import draft_responses
from find_meeting_time import find_meeting_time
from rewrite import rewrite
from config import get_config
from human_inbox import (
    send_message,
    send_email_draft,
    notify,
    send_cal_invite
)
from gmail import (
    send_email,
    mark_as_read,
    send_calendar_invite
)
from models import State

class ConfigSchema(TypedDict):
    db_id: int
    model: str

def human_node(state: State):
    pass

graph_builder = StateGraph(State, config_schema=ConfigSchema)
graph_builder.add_node(human_node)
graph_builder.add_node(triage_input)
graph_builder.set_entry_point("triage_input")
graph_builder.add_node(draft_responses)
graph_builder.add_node(send_message)
graph_builder.add_node(rewrite)
graph_builder.add_node(mark_as_read_node)
graph_builder.add_node(send_email_draft)
graph_builder.add_node(send_email)
graph_builder.add_node(send_email_node)
graph_builder.add_node(bad_tool_name)
graph_builder.add_node(send_calendar_invite)
graph_builder.add_node(send_calendar_invite_node)
graph_builder.add_node(find_meeting_time)
graph_builder.add_node(notify)