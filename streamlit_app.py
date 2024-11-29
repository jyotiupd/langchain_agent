
import streamlit as st
from langchain.agents import AgentType, Tool, initialize_agent, AgentExecutor, create_openai_tools_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils.agents_prompt_config import tools, agent_prompt,memory
from utils.load_model import embeddings, model
#
agent = create_openai_tools_agent(model, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,  handle_parsing_errors = True, verbose=False)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)
#
def process_text(user_input):
    print("processing text")
    result = agent_with_chat_history.invoke({"input": user_input,"tools":tools},config={"configurable": {"session_id": "Santosh"}})
    print(result['output'])
    return result['output']
#
st.title("Sample Chatbot")
if 'history' not in st.session_state:
    st.session_state.history = []
user_input = st.text_input("Please enter your query below and click enter.",key="user_input")

# Generate response and update history
##sample_input :- How many leaves do employee '39592' have? and divide with the balance insurance amount from total leaves of this employee. provide me each answer in new line.
if st.button("Send"):
    if user_input:
        print("Input received")
        bot_response = process_text(user_input)
        st.session_state.history.append((user_input, bot_response))


# Display conversation
chat_text = ""
for user_msg, bot_msg in st.session_state.history:
    chat_text += f"You: {user_msg}\nBot: {bot_msg}\n\n"
st.text_area("Chat", value=chat_text, height=300, disabled=False)