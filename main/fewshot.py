from langgraph.store.base import BaseStore
from models import Email

email_template = """
Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content:

```
{content}

```
> Triage Result: {result}
"""

def format_similar_examples_store(examples):
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            email_template.format(
                subject = eg.value["input"]["subject"],
                from_email= eg.value["input"]["from_email"],
                to_email= eg.value["input"]["to_email"],
                content= eg.value["input"]["page_content"][:400],
                result= eg.value["triage"],
            )
        )
    
    return "\n\n-----------------------\n\n".join(strs)

async def get_few_shot_examples(email:Email, store:BaseStore, config):
    namespace = (config["configurable"].get("assistant_id","default"), "triage_examples")
    result = await store.asearch(namespace, query=str({"input": "email"}), limit=5)
    
    return "" if result in None else format_similar_examples_store(result)

