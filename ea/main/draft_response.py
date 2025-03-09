from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore
from ea.main.models import (
    State,
    NewEmailDraft,
    ResponseEmailDraft,
    Question,
    MeetingAssistant,
    SendCalendarInvite,
    Ignore,
    email_template
)
from ea.main.config import get_config


EMAIL_WRITING_INSTRUCTIONS = """You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.

{background}

{name} gets lots of emails. This has been determined to be an email that is woth {name} responding to.

Your job is to help {name} respond. You can do this in a few ways.

# Using the `Question` tool

First, get all the required information to respond. You can use the Question tool to ask {name} for information if you do not know it.

When drafting emails either to respond on thread or, if you do not have all the information needed to respond in appropriate way, call the `Question` tool until you have that information. Do not put placeholders for names or emails or information - get that directly from {name}!
You can get this information by calling the `Question` tool. Again - Do not under any circumstances, draft an email with placeholders or you will get fired.

If people ask {name} if they can attend some event or meet with them, do not agree to do so unless {name} has explicitly approved of it to you.

Remember, if you don't have the needed information, you can always ask {name} for it by using the `Question` tool.
Never just make things up! So if you do not know something, or don't know what {name} would prefer, don't hesitate to ask {name}.
Never use `Question` tool to ask {name} when they are free. Instead, just ask the MeetingAssistant.

# Using the `ResponseEmailDraft` tool

Next, if you have enough information to respond, you can draft an email for {name}. Use the `ResponseEmailDraft` tool for this.

ALWAYS draft email as if they are coming from {name}. Never draft emails as {name}'s assitant or someone else.

When adding new recipients - only do so if {name} explicitly asks for it and only if you know their email ids. If you don't know the right email ids to add in, then ask {name}. You do NOT need to add in people who are already on the email! Do not make up emails! - this is very important!

{response_preferences}

# Using the `SendCalendarInvite` tool

Sometimes you will want to schedule a calendar event. You can do this with `SendCalendarInvite` tool.
If you know that {name} would want to schedule a meeting, and if you know that {name}'s calendar is free for that period, you can schedule a meeting with `SendCalendarInvite` tool.
Once you decide to schedule a meeting, you should always confirm with {name} about the time of the meeting before sending the invites to others.

{schedule_preferences}

# Using the `NewEmailDraft` tool

Sometimes you will need to start a new email thread. If you have all the necessary information for this, use the `NewEmailDraft` tool to do it.

If {name} asks someone if it's okay to introduce them, and they respond yes, you should draft a new email with that introduction.

# Using the `MeetingAssistant` tool

If the request for meeting is from a legitimate person and is working to schedule a meeting, call the `MeetingAssistant` tool to get a response from a specialist!
You should not ask {name} for meeting time, unless the MeetingAssistant is unable to find a way.
If they ask for times from {name}, first ask the MeetingAssistant by calling the `MeetingAssistant` tool.
Note that you should only call this if working to schedule a meeting - If a meeting has already been scheduled, and they're referencing it, then no need to call this tool.

# Background information: information you might find helpful when responding to emails or deciding what to do.

{other_preferences}
"""
########################################################################################################################################################################################################################################################################################################

draft_prompt = """{instructions}

Remember to call this tool correctly! Use the specified names exactly - not add `functions::` to the start. Pass all the required arguments.

Here is the email thread. Note that this is the full email thread. Pay special attention to the most recent email.

{email}"""

########################################################################################################################################################################################################################################################################################################

#TODO: can be split into 2 functions


async def draft_responses(state:State, config: RunnableConfig, store:BaseStore):
    """Write an email"""
    model = config["configurable"].get("model", "gpt-4o")
    llm = ChatOpenAI(
        model = model,
        temperature = 0,
        parallel_tool_calls = False, #default is True
        tool_choice = "required"
    )
    tools = [
        NewEmailDraft,
        ResponseEmailDraft,
        Question,
        MeetingAssistant,
        SendCalendarInvite
    ]
    messages = state.get("messages", [])
    if len(messages) > 0: #thisi is so that, the first time if i decide to ignore the mail, it can just ignore this email instead of trying to draft a response.
        tools.append(Ignore)
    prompt_config = get_config(config)
    namespace = (config["configurable"].get("assistant_id", "default"))
    
    key = "schedule_preferences"
    result = await store.aget(namespace, key)
    if result and "data" in result.value:
        schedule_preferences = result.value["data"]
    else:
        await store.aput(namespace, key, {"data":prompt_config["schedule_preferences"]})
        schedule_preferences = prompt_config["schedule_preferences"]
    
    key = "random_preferences"
    result = await store.aget(namespace, key)
    if result and "data" in result.value:
        random_preferences = prompt_config[key]
    else:
        await store.aput(namespace, key, {"data": prompt_config["background_preferences"]})
        random_preferences = prompt_config["background_preferences"]
    
    key = "response_preferences"
    result = await store.aget(namespace, key)
    if result and "data" in result.value:
        response_preferences = prompt_config[key]
    else:
        await store.aput(namespace, key, {"data": prompt_config[key]})
        response_preferences = prompt_config[key]
    
    _prompt = EMAIL_WRITING_INSTRUCTIONS.format(
        schedule_preferences = schedule_preferences,
        random_preferences = random_preferences,
        response_preferences = response_preferences,
        name = prompt_config["name"],
        full_name = prompt_config["full_name"],
        background = prompt_config["background"]
    )
    input_message = draft_prompt.format(
        instructions = _prompt,
        email = email_template.format(
            email_thread = state["email"]["page_content"],
            author = state["email"]["from_email"],
            to = state["email"].get("to_email", ""),
            subject = state["email"]["subject"]
        )
    )

    model = llm.bind_tools(tools)
    messages = [{"role":"user", "content": input_message}] + messages
    #the tool choice = required. For some reason if the model doesn't call the tool, then we retry up to 5 times, and manually add a user message asking the model to make a tool call.
    i = 0
    while i < 5:
        response = await model.ainvoke(messages)
        if len(response.tool_calls) != 1:
            i += 1
            messages += [{"role": "user", "content": "Please call a valid tool call."}]
        else:
            break
        
        return {"draft": response, "messages": [response]}
