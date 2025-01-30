from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.graph.message import AnyMessage
from langgraph.graph import add_messages


class Email(TypedDict):
    id: str
    thread_id: str
    from_email: str
    subject: str
    page_content: str
    send_time: str
    to_email: str

class RespondTo(BaseModel):
    logic: str = Field(description="Reason for the response to be the way it is")
    response: Literal["no", "email", "notify", "question"] = "no"

class ResponseEmailDraft(BaseModel):
    """Draft of an email to send as a response"""
    content: str
    recipients: List[str]

class ReWriteEmail(BaseModel):
    """Logic for rewriting an email"""
    tone_logic: str = Field(description = "Logic for what the tone of the rewritten email should be ")

class NewEmailDraft(BaseModel):
    """Draft of a new email to send"""
    content: str
    recipients: List[str]

class Question(BaseModel):
    """Question to ask user"""
    content: str

class Ignore(BaseModel):
    """Call this to ignore the email. Only call this if the user specifies to ignore the email"""
    ignore: str

class MeetingAssistant(BaseModel):
    """Call this to have user's meeting assistant look at it"""
    call: bool

class SendCalendarInvite(BaseModel):
    """Call this to send a calendar invite"""
    emails: List[str] = Field(description="List of emails to send calendar invites for. Do NOT make any emails up!")
    title: str = Field(description="Name of the meeting")
    start_time: str = Field(description="Start time of the meeting. Should be in '2025-01-28T21:49:00' format")
    end_time: str = Field(description="End time of the meeting. Should be in '2025-01-28T21:49:00' format")



def convert_obj(o, m):
    if isinstance(m, dict):
        return RespondTo(**m)
    else:
        return m


class State(TypedDict):
    email: Email
    triage: Annotated[RespondTo, convert_obj]
    messages: Annotated[List[AnyMessage], add_messages]



email_template = """From: {author}
To: {to}
Subject: {subject}

{email_thread}"""