import streamlit as st
import time

# --- Mock function for model response ---
# Replace this with your Jira + LangGraph + LLM pipeline call
def get_model_response(question):
    # Example response from your AI model
    return "The latest update on ticket ABC-123 is that it is currently assigned to John Doe and awaiting QA approval."

# --- Streamlit UI ---
st.set_page_config(page_title="Jira Smart Assistant", page_icon="🤖")

st.title("🤖 Jira Smart Assistant")
st.write("Ask me anything about your Jira tickets.")

# User input
user_question = st.text_input("Enter your question:", placeholder="e.g., What's the latest update on ABC-123?")

# Submit button
if st.button("Ask"):
    if user_question.strip():
        # Get model's full answer
        full_answer = get_model_response(user_question)
        
        # Simulate ChatGPT-style word-by-word display
        placeholder = st.empty()
        displayed_text = ""
        for word in full_answer.split():
            displayed_text += word + " "
            placeholder.markdown(displayed_text)
            time.sleep(0.05)  # delay for streaming effect
    else:
        st.warning("Please enter a question.")
