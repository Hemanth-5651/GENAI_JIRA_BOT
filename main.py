import torch,os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from jira import JIRA
from dotenv import load_dotenv
import json
load_dotenv()
print(os.getenv("JIRA_URL"))
print(os.getenv("JIRA_USER"))
print(os.getenv("JIRA_API_TOKEN"))
jira = JIRA(
    server=os.getenv("JIRA_URL"),
    basic_auth=(os.getenv("JIRA_USER"), os.getenv("JIRA_API_TOKEN"))
)
def get_issue_details(ticket_id: str):
    issue = jira.issue(ticket_id)
    print('hi')
    # Extract comments
    comments = []
    for c in issue.fields.comment.comments:
        comments.append({
            "author": c.author.displayName,
            "body": c.body,
            "created": c.created
        })

    # Extract sprint info (if present)
    sprint_info = None
    if hasattr(issue.fields, 'customfield_10020'):  # common sprint custom field
        if issue.fields.customfield_10020:
            sprint_info = [sprint.name for sprint in issue.fields.customfield_10020]

    # Extract team name (if custom field exists)
    team_name = None
    if hasattr(issue.fields, 'customfield_10021'):  # replace with actual team name field ID
        team_name = issue.fields.customfield_10021

    # Build dictionary
    details = {
        "id": issue.id,
        "key": issue.key,
        "summary": issue.fields.summary,
        "description": issue.fields.description,
        "status": issue.fields.status.name,
        "priority": issue.fields.priority.name if issue.fields.priority else None,
        "assignee": issue.fields.assignee.displayName if issue.fields.assignee else None,
        "reporter": issue.fields.reporter.displayName if issue.fields.reporter else None,
        "labels": issue.fields.labels,
        "created": issue.fields.created,
        "updated": issue.fields.updated,
        "comments": comments,
        "sprint": sprint_info,
        "team_name": team_name
    }

    return details
print(get_issue_details("SCRUM-1"))
# # --- Load Gemma 2 2B ---

tokenizer_gemma = AutoTokenizer.from_pretrained("gemma_model_4bit")

model_gemma = AutoModelForCausalLM.from_pretrained(
    "gemma_model_4bit",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
    llm_int8_enable_fp32_cpu_offload=True
)
print(f"Gemma 2 2B loaded")


model=model_gemma
tokenizer=tokenizer_gemma
ticket_data=get_issue_details("SCRUM-1")
ticket_context = "\n".join(f"{k}: {v}" for k, v in ticket_data.items())

# 3. Create pipeline
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# 4. Ask a question about the ticket
question = "What is the issue and what was the latest status"
prompt = f"""You are a helpful assistant. Here is the JIRA ticket info:

{ticket_context}

Now answer the question: {question}
"""

# 5. Generate answer
response = chatbot(prompt, max_new_tokens=200, temperature=0.2)
print(response[0]["generated_text"])
model_gemma.save_pretrained("gemma_model_4bit")

# Save tokenizer (important for reloading)
tokenizer_gemma.save_pretrained("gemma_model_4bit")

