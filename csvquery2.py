import streamlit as st
import pandas as pd      
from langchain_openai import ChatOpenAI  
from langchain_community.agent_toolkits import SQLDatabaseToolkit  
from langchain_core.messages import HumanMessage  
from langgraph.prebuilt import create_react_agent  
from langchain_community.utilities import SQLDatabase          
from sqlalchemy import create_engine    
import os
import tempfile
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="AI Data Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 1rem 0;
    }
    .chat-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .quick-action-btn {
        background: linear-gradient(135deg, #ff6b6b, #ee5a6f);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        border: none;
        font-weight: bold;
        margin: 0.2rem;
        cursor: pointer;
    }
    .sidebar-content {
        background: #2c3e50;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_db_from_file(file_path, table_name=None):
    """
    Create SQLite database from CSV or Excel file.
    """
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    if table_name is None:
        table_name = filename.lower().replace(' ', '_').replace('-', '_')
    
    db_filename = f"{filename}.db"
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    engine = create_engine(f"sqlite:///{db_filename}")
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    db = SQLDatabase(engine=engine)
    
    return db, df

def initialize_agent(db):
    """Initialize the SQL agent"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    system_message = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct SQLite query to run,
    then look at the results of the query and return the answer. Unless the user
    specifies a specific number of examples they wish to obtain, always limit your
    query to at most 5 results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.

    You MUST double check your query before executing it. If you get an error while
    executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    database.

    To start you should ALWAYS look at the tables in the database to see what you
    can query. Do NOT skip this step.

    Then you should query the schema of the most relevant tables.
    """
    
    agent_executor = create_react_agent(llm, tools, prompt=system_message)
    return agent_executor

def query_agent(agent, question):
    """Query the agent with a question"""
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        return result["messages"][-1].content
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'question_counter' not in st.session_state:
    st.session_state.question_counter = 0

# Sidebar
with st.sidebar:
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("📊 Data Analysis", key="data_analysis"):
        st.session_state.quick_action = "analysis"
    
    if st.button("🤖 AI Powered", key="ai_powered"):
        st.session_state.quick_action = "ai"
    
    if st.button("⚡ Instant Results", key="instant_results"):
        st.session_state.quick_action = "instant"
    
    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")
    st.markdown("Drag and drop or click to select")
    st.markdown("Limit 200MB per file • CSV, XLS, XLSX")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel files"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            db, df = create_db_from_file(tmp_file_path)
            st.session_state.db = db
            st.session_state.df = df
            st.session_state.agent = initialize_agent(db)
            st.success(f"✅ File uploaded successfully!")
            st.info(f"📊 {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
        finally:
            os.unlink(tmp_file_path)
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Results")
    st.markdown("• Ask specific questions about your data")
    st.markdown("• Use natural language")
    st.markdown("• Be clear about what you want to know")

# Main content
st.markdown("""
<div class="main-header">
    <h1>🧠 AI Data Assistant</h1>
    <p>Transform your CSV and Excel data into intelligent conversations</p>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Smart Analysis</h3>
        <p>AI-powered insights from your data</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>💬 Natural Queries</h3>
        <p>Ask questions in plain English</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>⚡ Lightning Fast</h3>
        <p>Get instant answers to your questions</p>
    </div>
    """, unsafe_allow_html=True)

# Main interaction area
if st.session_state.db is None:
    st.markdown("""
    <div class="upload-section">
        <h2>🚀 Ready to Get Started?</h2>
        <p>Upload your CSV or Excel file using the sidebar and start asking questions about your data!</p>
        <br>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <button class="quick-action-btn">📊 Data Analysis</button>
            <button class="quick-action-btn">🤖 AI Powered</button>
            <button class="quick-action-btn">⚡ Instant Results</button>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Data overview
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(st.session_state.df))
            st.metric("Total Columns", len(st.session_state.df.columns))
        with col2:
            st.write("**Column Names:**")
            st.write(", ".join(st.session_state.df.columns.tolist()))
        
        st.write("**Data Preview:**")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    # Chat interface
    st.markdown("### 💬 Ask Questions About Your Data")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**You:** {question}")
            st.markdown(f"**AI Assistant:** {answer}")
            st.markdown("---")
    
    # Input area
    user_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the top 5 sales by region?",
        key=f"user_input_{st.session_state.question_counter}"
    )
    ask_button = st.button("Ask", type="primary", use_container_width=False)
    
    # Process question
    if ask_button and user_question and st.session_state.agent:
        with st.spinner("🤔 Thinking..."):
            answer = query_agent(st.session_state.agent, user_question)
            st.session_state.chat_history.append((user_question, answer))
            # Increment counter to create new input widget
            st.session_state.question_counter += 1
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>AI Data Assistant • Transform your data into insights •</p>
</div>
""", unsafe_allow_html=True)
