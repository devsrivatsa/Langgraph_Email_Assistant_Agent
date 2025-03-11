from datetime import datetime
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from ea.gmail import get_events_for_days
from ea.main.models import State
from ea.main.config import get_config

meeting_prompt = """You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.

The below email thread has been flagged as requesting time to meet. Your sole purpose is to survey {name}'s calendar and schedule meetings for {name}.

If the email is suggesting some specific times, then check if {name} is available then.

If the email asks for time, use the tool to find valid times to meet (always suggest them in {time_zone}).

If they express preferences in their email thread, try to abide by those. Do not suggest times they have already said won't work, or times outside of the options suggested by them.

Try to send available spots in as big of chunks as possible. For example, if {name} has 1pm - 3pm open, send:

```
1pm - 3pm 
```
NOT

```
1-1:30pm
1:30-2pm
2-2:30pm
2:30-3pm
```

Do not send time slots less than 15 minutes in length.

Your response should be extremely high density. You should not respond directly to the email, but rather just say factually whether {name} is free, and at what time slots. Do not give any extra commentary. Examples of good responses include:

<examples>

Example 1:

> {name} is free 9:30 - 10

Example 2:

> {name} is not free then. But {name} is free at 10:30.

</examples>

The current date is {current_date}

Here is the email thread:

From: {author}
Subject: {subject}

{email_thread}
"""

async def find_meeting_time(state:State, config:RunnableConfig):

    model = config["configurable"].get("model", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0)
    agent = create_react_agent(llm, [get_events_for_days])

    current_date = datetime.now()

    prompt_config = get_config(config)

    input_message = meeting_prompt.format(
        email_thread = state["email"]["page_content"],
        author = state["email"]["from_email"],
        subject = state["email"]["subject"],
        current_date = current_date.strftime("%A %B %d, %y"),
        name = prompt_config["name"],
        full_name = prompt_config["full_name"],
        time_zone = prompt_config["timezone"]
    )

    messages = state.get("messages", [])
    messages = messages[:-1] #we do this because there is just a tool call for routing

    result = await agent.ainvoke({"messages": [{"role": "user", "content": input_message}] + messages})
    prediction = state["messages"][-1]
    tool_call = prediction.tool_calls[0]
    
    return {
        "messages": [
            ToolMessage(
                content = result["messages"][-1].content, 
                tool_call_id=tool_call["id"]
            )
        ]
    }