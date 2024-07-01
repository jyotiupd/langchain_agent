#
import warnings
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent, AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
#
from utils.load_model import embeddings, model # Replace or use your model here
#
warnings.filterwarnings('ignore')

erc_stuff_prompt_override = """

Role: Experienced company representative, proficient at finding information pertinent to any employee 
queries in your employee policies.
Task: Your task is to respond to the User Query with detailed information using the ``context``  and 
``metadata`` provided below.
Tone: Respond to the user query in a professional, courteous, and helpful tone
Adhere strictly to the company's policies and guidelines, and ensure your response is informative, 
accurate, and formatted according to the provided template.
-----
Context: {context}
-----
Template for Response:
Answer: [Provide a clear and direct answer to the user's inquiry.]
Explanation: [Provide a detailed explanation of the company's policy or guideline, using plain language and avoiding jargon.]
Source: [Reference the source of the information, including document names and page numbers.]

----------------
Given the Context and Instructions above, answer the below User Query with as much detail as possible.
-----
User Query: '{query}'
-----

"""
#
agent_prompt_template = """
    You are an exceptionally helpful assistant dedicated to providing accurate answers to user queries. 
    
    You have access to only specific set of tools each designed to assist users effectively and answer to those queries which are related to given tools, here are the tools you have {tools}. 
    Also provide answers which were there in chat history if it is related.
    If you are unable to provide information realted to the user query from the given tools simply respond with "I don’t know, my information is limited only with leaves, insurance, IT and policies.", avoid making up answers. 
    Here is the input {input} for which you need to provide an answer
    """
#
memory = ChatMessageHistory(messages=[])
#
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", agent_prompt_template),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
#
@tool
def get_policy_details(query: str):
    """
    A function that retrieves a policy answer based on a query using a specified index and model.

    Parameters:
    query (str): The query string used to search for the policy answer.
    eid (Optional): The entity ID to filter the search by.
    e_name (Optional): The entity name to filter the search by.

    Returns:
    str: The policy answer retrieved based on the query.
    """
    print(f"INFORM--Inside get_policy_answer: {query}")
    #
    index_path = "./data/erc_docs"
    policy_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    #
    retriever = policy_db.as_retriever(search_type="mmr", search_kwargs={"fetch_k": 20, "k": 5})
    #
    docs = retriever.get_relevant_documents(query=query, verbose=False)
    #
    erc_prompt = PromptTemplate(template=erc_stuff_prompt_override, input_variables=["context", "query"])
    #
    llm_chain = LLMChain(llm=model, prompt=erc_prompt)
    #
    chain = StuffDocumentsChain(
        llm_chain=llm_chain, 
        document_prompt=PromptTemplate(input_variables=["page_content"], template="{page_content}"),
        document_variable_name="context",
    )
    result = chain.run(input_documents=docs, query=query)
    #
    return docs, result
#
# SQL Toolkit initializations
leave_db = SQLDatabase.from_uri("sqlite:///data/db/employee_leaves.db")
insurance_db = SQLDatabase.from_uri("sqlite:///data/db/health_insurance.db")
it_db = SQLDatabase.from_uri("sqlite:///data/db/IT_support.db")
#
leave_toolkit = SQLDatabaseToolkit(db=leave_db, llm=model)
ins_toolkit = SQLDatabaseToolkit(db=insurance_db, llm=model)
it_toolkit = SQLDatabaseToolkit(db=it_db, llm=model)
#
leave_agent_executor = create_sql_agent(
    llm=model, toolkit=leave_toolkit, verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={"handle_parsing_errors": True}
)
insurance_agent_executor = create_sql_agent(
    llm=model, toolkit=ins_toolkit, verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={"handle_parsing_errors": True}
)
it_agent_executor = create_sql_agent(
    llm=model, toolkit=it_toolkit, verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={"handle_parsing_errors": True}
)
#
tools = [
    Tool(
        name="Leave_Agent_System",
        func=leave_agent_executor.run,
        description="useful for when you need to answer questions about the leaves of an employee",    
    ),
    Tool(
        name="Insurance_Agent_System",
        func=insurance_agent_executor.run,
        description="useful for when you need to answer questions about insurance details of an employee",    
    ),
    Tool(
        name="IT_Support_system",
        func=it_agent_executor.run,
        description="useful for when you need to answer questions about IT tickets. Input should be a fully formed question.",   
    ),
    Tool(
        name="Get_Policy_Details",
        func=get_policy_details,
        description="A tool that returns textual information about any policies",
    ),
]
#
