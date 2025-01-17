import streamlit as st
import ast
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain import hub
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver



# Initialize SQL database
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Read table schema
def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except (FileNotFoundError, IOError) as e:
        return f"Error reading file: {e}"

filepath = "table_info.txt"
table_info = read_file(filepath)

# Define LangChain LLM
llm = ChatOllama(model="llama3.2", temperature=0)

# Define prompt template
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given an input question, create a syntactically correct {dialect} query to run to help find the answer (Only Generate the query and don't add (```sql) to the output and you must check the tables columns and if the colunm you looking for is not present you can try to calculate it from other columns before generating the query and dont use unnecessary join)."
                "Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query results You Must add LIMIT to the query with ({top_k}). You can order the results by a relevant column to return the most interesting examples in the database."
                "\n\nNever query for all the columns from a specific table, only ask for a the few relevant columns given the question.\n\nPay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist."
                "Also, pay attention to which column is in which table.\n\nOnly use the following tables:\n{table_info}\n\nQuestion: {input}"
            ),
            (
                "human", "{input}"
            ),
        ]
    )
    rag_chain = prompt | llm
    response = rag_chain.invoke({
        "dialect": db.dialect,
        "top_k": 10,
        "table_info": table_info,
        "input": state["question"],
    })
    state["query"] = response.content
    return state

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    result = execute_query_tool.invoke(state["query"])
    state["result"] = result
    return state

def extract_answer(state: State):
    """Extract all answers from the query result."""
    try:
        parsed_result = ast.literal_eval(state["result"])
        answers = [", ".join(map(str, item)) for item in parsed_result]
        state["answer"] = answers
    except Exception as e:
        state["answer"] = [f"Error extracting answers: {e}"]
    return state

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question in a one line.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response}

def validate_and_correct_query(state: State):
    """Validate and correct the SQL query using LLM."""
    validation_prompt = (
        "You are a SQL expert. Given the following SQL query and database schema, validate and correct the query if necessary. "
        "Culculate new columns if not exist using math operations based on existing columns if necessary. "
        "Ensure the query is syntactically correct and adheres to the schema. If the query is valid, return it as is. "
        "If it requires corrections, provide the corrected query. Return only the query without explanation.\n\n"
        f"Schema: {table_info}\n\n"
        f"SQL Query: {state['query']}"
    )

    response = llm.invoke(validation_prompt)
    state["query"] = response.content.strip()
    return state

# Update the workflow to include the validation step
graph_builder = StateGraph(State).add_sequence([
    write_query,
    validate_and_correct_query,  # New validation step
    execute_query,
    generate_answer,
])

graph_builder.add_edge(START, "write_query")

# Recompile the graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Configuration for LangGraph execution
config = {"configurable": {"thread_id": "streamlit_user_session"}}

# Streamlit app setup
st.title("SQL Agent with Llama 3.2 3B & Ollama")
st.divider()
st.write("Ask a question, generate an SQL query, and execute it against the Chinook database.")
st.write("Tools: Ollama - Langchain - LangGraph - SQLite")

st.image("llama.jpg")

# Question input from user
question = st.text_input("Enter your question:", placeholder="How many Employees are there?")

# Generate query and execute it using LangGraph
if st.button("Generate and Execute Query"):
    if question:
        # Initialize state
        state = State(question=question, query="", result="", answer="")

        # Execute graph with thread ID
        st.write("**Executing workflow...**")
        try:
            for step in graph.stream(state, config=config):
                for stage_name, stage_output in step.items():
                    st.write(f"**Stage: {stage_name} completed.**")

                # Update state dynamically during the workflow
                state.update(stage_output)

        except Exception as e:
            st.error(f"An error occurred during workflow execution: {e}")

        st.image(graph.get_graph().draw_mermaid_png())
        st.divider()

        # Display final results
        st.markdown(f"### **Question**\n{state['question']}")
        st.divider()
        st.markdown(f"### **Generated Query**\n```sql\n{state['query']}\n```")
        st.divider()

        # Parse and display query result
        if state["result"]:
            try:
                # Parse the result into a readable format
                parsed_result = ast.literal_eval(state["result"])
                if isinstance(parsed_result, list) and len(parsed_result) > 0 and isinstance(parsed_result[0], tuple):
                    st.write("**Query Result:**")
                    st.table(parsed_result)  # Display as a table if structured
                else:
                    st.write("**Query Result:**")
                    st.write(parsed_result)  # Fallback for unstructured results
            except Exception as e:
                st.error(f"Error parsing query result: {e}")
        else:
            st.write("No results found.")

        st.divider()

        # Display extracted answers
        if state["answer"]:
            st.write("**Final Answer:**")
            st.markdown(f"- **{state['answer'].content}**")
        else:
            st.write("No answers extracted.")

    else:
        st.error("Please enter a question before generating the query.")
