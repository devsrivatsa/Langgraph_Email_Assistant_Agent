from typing import TypedDict
from ea.gmail import fetch_group_emails
from langgraph_sdk import get_client
import httpx
import uuid
import hashlib
from langgraph.graph import StateGraph, START, END
from ea.main.config import get_config

client = get_client()


class JobKickoff(TypedDict):
    minutes_since: int


async def main(state: JobKickoff, config):
    minutes_since: int = state["minutes_since"]
    email = get_config(config)["email"]

    # TODO: This really should be async
    for email in fetch_group_emails(email, minutes_since=minutes_since):
        thread_id = str(
            uuid.UUID(hex=hashlib.md5(email["thread_id"].encode("UTF-8")).hexdigest())
        )
        try:
            thread_info = await client.threads.get(thread_id)
        except httpx.HTTPStatusError as e:
            if "user_respond" in email:
                continue
            if e.response.status_code == 404:
                thread_info = await client.threads.create(thread_id=thread_id)
            else:
                raise e
        if "user_respond" in email:
            await client.threads.update_state(thread_id, None, as_node="__end__")
            continue
        recent_email = thread_info["metadata"].get("email_id")
        if recent_email == email["id"]:
            break
        await client.threads.update(thread_id, metadata={"email_id": email["id"]})

        await client.runs.create(
            thread_id,
            "main",
            input={"email": email},
            multitask_strategy="rollback",
        )


crong_graph_builder = StateGraph(JobKickoff)
crong_graph_builder.add_node(main)
crong_graph_builder.add_edge(START, "main")
crong_graph_builder.add_edge("main", END)
cron_graph = crong_graph_builder.compile()