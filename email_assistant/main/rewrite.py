"""Agent responsible for rewriting the response in more appropriate tone"""

from langchain_openai import ChatOpenAI
from models import State, ReWriteEmail
from config import get_config


rewrite_prompt = """Your job is to rewrite an email draft to sound more like {name}

{name}'s assistant just drafted an email. It is factualy correct, but it may not sound like {mame}. \
Your job is to rewrite the email keeping the information the same (do not add anything else that is made up!) \
but adjusting the tone.

{instructions}

Here is the assistant's current draft:

<draft>
{draft}
</draft>

Here is the email thread:

From: {author} 
To {to}:
Subject: {subject}

{email_thread}"""

async def rewrite(state: State, config, store):
    model = config["configurable"].get("model", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0)
    prev_msg = state["messages"][-1]
    draft = prev_msg.tool_calls[0]["args"]["content"]
    namespace = (config["configurable"].get("assistant_id", "default"),)
    result = await store.aget(namespace, "rewrite_instructions")
    prompt_config = get_config(config)
    if result and "data" in result.value:
        _prompt = result.value["data"]
    else:
        await store.aput(
            namespace, 
            "rewrite_instructions", 
            {"data": {"instructions": ""}}
        )
        _prompt = prompt_config["rewrite_preferences"]
    input_message = rewrite_prompt.format(
        name=prompt_config["name"],
        instructions=_prompt,
        draft=draft,
        email_thread=state["email"]["page_content"],
        author=state["email"]["from_email"],
        to=state["email"]["to_email"],
        subject=state["email"]["subject"]
    )
    model = llm.with_structured_output(ReWriteEmail).bind(
        tool_choice={"type": "function", "function": {"name": "ReWriteEmail"}}
    )
    response = await model.ainvoke(input_message)

    tool_calls = [
        {
            "id": prev_msg.tool_calls[0]["id"],
            "name": prev_msg.tool_calls[0]["name"],
            "args": {
                **prev_msg.tool_calls[0]["args"],
                **{"content": response.rewritten_content}
            }
        }
    ]

    prev_msg = {
        "role": "assistant",
        "id": prev_msg.id,
        "content": prev_msg.content,
        "tool_calls": tool_calls
    }

    return {"messages": [prev_msg]}