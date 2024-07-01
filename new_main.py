
#
from datasets import Dataset 
from ragas.metrics.critique import harmfulness
from ragas import evaluate
#
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
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)
#
def process_text(user_input):
    docs = []
    result = ""
    result = agent_with_chat_history.invoke({"input": user_input,"tools":tools},config={"configurable": {"session_id": "Santosh"}})
    #
    # data_samples = {
                        # 'question': user_input,
                        # 'answer': result,
                        # 'contexts' : [docs],
                    # }
    # dataset = Dataset.from_dict(data_samples)
    # score = evaluate(dataset,metrics=[harmfulness])
    
    # print ("INFORM--Socre--", score)

    return result['output']
#
st.title("Sample Chatbot")
user_input = st.text_input("Please enter your query below and click enter.")
 
if user_input:
  processed_text = process_text(user_input)
  st.write("Output:", processed_text)