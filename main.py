import json
import os
import re
import warnings

# -------- env_chatbot ------------
# .. this contains latest libraries use this.
#
#import gradio as gr
from dotenv import load_dotenv
#from langchain.agents.agent_types import AgentType
from langchain.agents import AgentType
from langchain.chains import LLMChain, StuffDocumentsChain
#from langchain.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
#from langchain.vectorstores.faiss import FAISS
from langchain_community.vectorstores import FAISS
#from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
#
# -------- env_chatbot ------------
#
# -------- env_new_chatbot --------
#
# from dotenv import load_dotenv
# from langchain.agents.agent_types import AgentType
# from langchain.chains import LLMChain, StuffDocumentsChain
# from langchain.sql_database import SQLDatabase
# from langchain.vectorstores.faiss import FAISS
# from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
#
# -------- env_new_chatbot --------
#
warnings.filterwarnings('ignore')

from utils.load_model import embeddings, model # Replace or use your model here

#_ = load_dotenv('utils/config.env') #use your env file here

index_path = "./data/erc_docs"
with open('./data/emp_details.json', 'r') as f:
    emp_details = json.load(f)

emp_id = '38433'
emp_name = emp_details[emp_id]

# Run custom stuff chain enterprise_rag_chain (erc)
erc_document_prompt = PromptTemplate(input_variables=["page_content"], template="{page_content}")
erc_document_variable_name = "context"

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

erc_prompt = PromptTemplate(template=erc_stuff_prompt_override, input_variables=["context", "query"])


def get_policy_answer(query: str, eid: str = None, e_name: str = None) -> str:
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
    policy_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = policy_db.as_retriever(
        search_type="mmr", search_kwargs={
            "fetch_k": 20,
            "k": 5}
    )
    docs = retriever.get_relevant_documents(query=query, verbose=False)
    llm_chain = LLMChain(llm=model, prompt=erc_prompt, )
    chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_prompt=erc_document_prompt,
        document_variable_name=erc_document_variable_name,
    )
    result = chain.run(input_documents=docs, query=query)
    return result


# SQL Toolkit initializations
leave_db = SQLDatabase.from_uri("sqlite:///data/db/employee_leaves.db")
insurance_db = SQLDatabase.from_uri("sqlite:///data/db/health_insurance.db")
it_db = SQLDatabase.from_uri("sqlite:///data/db/IT_support.db")

leave_toolkit = SQLDatabaseToolkit(db=leave_db, llm=model)
leave_agent_executor = create_sql_agent(
    llm=model, toolkit=leave_toolkit, verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

ins_toolkit = SQLDatabaseToolkit(db=insurance_db, llm=model)
insurance_agent_executor = create_sql_agent(
    llm=model, toolkit=ins_toolkit, verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

it_toolkit = SQLDatabaseToolkit(db=it_db, llm=model)
it_agent_executor = create_sql_agent(
    llm=model, toolkit=it_toolkit, verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={"handle_parsing_errors": True}
)


class QueryExecutor:
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor

    def execute_query_with_verification(self, query, e_id=None, e_name=None):
        """
        A method to execute a query, filter the database for the employee ID, and verify the employee name.
        If the retrieved record has a different name, it returns an error message.

        Parameters:
        - query (str): The query to be executed.
        - e_id (int): The employee ID to filter the database.
        - e_name (str): The name of the employee to verify the retrieved record.

        Returns:
        - str: The result of the query execution or an error message if the parsing fails.
        """
        try:
            # result = self.agent_executor.run(
                # f"filter the database for the employee id {e_id} and find the answer "
                # f"for {query} from the retrieved records. "
                # "Do not use the employee id in the final output. "
            # )
            #print (f"INFORM--Calling agent_executor-{self.agent_executor}")
            result = self.agent_executor.invoke(
                f"filter the database for the employee id {e_id} and find the answer "
                f"for {query} from the retrieved records. "
                "Do not use the employee id in the final output. "
            )
        except ValueError as e:
            result = str(e)
            if not result.startswith("Could not parse LLM output: `"):
                raise e
            result = result.removeprefix("Could not parse LLM output: `").removesuffix("`")
        return result


def get_leave_answer_sql(query, e_id, e_name):
    """
    A function to get the SQL query to filter the database for the employee ID and retrieve the answer if the
    employee name matches.
    """
    print ("INFORM--Calling get_leave_answer_sql function.")
    leave_executor = QueryExecutor(leave_agent_executor)
    return leave_executor.execute_query_with_verification(query, e_id, e_name)


def get_ins_answer_sql(query, e_id, e_name):
    """
    This function takes in a query, employee id, and employee name as parameters and retrieves the answer to the
    query from the insurance database.
    """
    insurance_executor = QueryExecutor(insurance_agent_executor)
    return insurance_executor.execute_query_with_verification(query, e_id, e_name)


def get_it_support_answer(query, e_id, e_name):
    """
    A function to get the IT support answer based on a query, employee ID, and employee name.
    """
    it_executor = QueryExecutor(it_agent_executor)
    return it_executor.execute_query_with_verification(query, e_id, e_name)


def extract_employee_id_from_query(query):
    """
    Extracts the employee ID from the given query.

    Parameters:
        query (str): The query from which to extract the employee ID.

    Returns:
        int or None: The extracted employee ID as an integer if found, None otherwise.
    """
    # Check if the query contains any of the keywords related to employee ID
    keywords = ['employee', 'employee id', 'emp id']
    if not any(keyword in query.lower() for keyword in keywords):
        return None

    # Use a regular expression to find a 5-digit number in the query
    pattern = r'\b\d{5}\b'
    match = re.search(pattern, query)
    if match:
        return int(match.group())
    else:
        return None


# Handle general questions.
gen_talk_ins = """You are a peppy, polite, and professional assistant skilled in small talk. Given the question:
{question}

Engage in small talk while consistently guiding the user to ask questions solely related to company policies, 
personal leaves, personal health insurance, and personal IT support ticket matters.
"""
custom_gen_talk = PromptTemplate(
    template=gen_talk_ins, input_variables=["question"]
)


def gen_talk(query: str) -> str:
    """
    A function that generates a response based on the provided query string.

    Parameters:
        query (str): The input query string.

    Returns:
        str: The generated response based on the query.
    """
    q_string = query
    llm_chain = ({
                     "question": RunnablePassthrough()} | custom_gen_talk | model | StrOutputParser())
    result = llm_chain.invoke(
        {
            "question": q_string}
    )
    print(f"printing result: {result}")
    return result


def get_context(query, eid=emp_id, e_name=emp_name):
    """
    A function that gets the context based on the query and employee details.

    Parameters:
        query (str): The query input.
        eid (int): The employee ID, default is emp_id.
        e_name (str): The employee name, default is emp_name.

    Returns:
        str: The context based on the query and employee details.
    """
    new_eid = extract_employee_id_from_query(query)

    if new_eid and new_eid != eid:
        return "You do not have access to other employee details."

    keyword_to_function = {
        'policy': get_policy_answer,
        'policies': get_policy_answer,
        'leaves': get_leave_answer_sql,
        'leave': get_leave_answer_sql,
        'health': get_ins_answer_sql,
        'insurance': get_ins_answer_sql,
        'ticket': get_it_support_answer
    }

    for keyword, func in keyword_to_function.items():
        if keyword in query:
            print (f"INFORM--keyword-{keyword}, function-{func}")
            if keyword != 'policies' and keyword != 'policy':
                return func(query, eid, e_name)
            else:
                return func(query, eid, e_name)

    return gen_talk(query)


query = "does Northwind standard policy cover dental benefits?"
query = "what is the data privacy policy?"
# query = "what is the status of my last ticket"
# query = "How many leaves do I have available?"
# query = "what is my health insurance name?"

print (get_context(query))
