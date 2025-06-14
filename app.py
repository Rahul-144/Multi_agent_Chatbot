import streamlit as st
import pandas as pd
import os
import re
import gc
import psutil
from typing import List, TypedDict, Dict, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
import json

# Page config
st.set_page_config(
    page_title="Multi-Agent RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    
    .chat-container {
        height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin-bottom: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 2px 10px rgba(240, 147, 251, 0.3);
        animation: slideInLeft 0.3s ease-out;
    }
    
    .system-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 10%;
        text-align: center;
        box-shadow: 0 2px 10px rgba(79, 172, 254, 0.3);
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .input-container {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        margin-top: 1rem;
    }
    
    .agent-badge {
        display: inline-block;
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        box-shadow: 0 2px 5px rgba(255, 107, 107, 0.3);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .agent-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 107, 107, 0.4);
    }
    
    .agent-badge.selected {
        background: linear-gradient(45deg, #38ef7d, #11998e);
        transform: translateY(-2px);
    }
    
    .memory-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 1rem 1.5rem;
        font-size: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    .agent-buttons-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1rem;
        justify-content: center;
    }

    .agent-button {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 1rem;
        font-weight: 500;
    }

    .agent-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 107, 107, 0.4);
    }

    .agent-button.selected {
        background: linear-gradient(45deg, #38ef7d, #11998e);
    }
</style>
""", unsafe_allow_html=True)

# Define state (same as original)
class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_agent: Optional[str]
    available_agents: List[str]

# PersonaRetriever class (same as original)
class PersonaRetriever:
    def __init__(self, csv_path: str, max_rows_per_persona: int = 1000, 
                 max_personas: int = 10, chunk_size: int = 10000):
        self.max_rows_per_persona = max_rows_per_persona
        self.max_personas = max_personas
        self.chunk_size = chunk_size
        
        # Load CSV with memory optimization
        self.df = self._load_csv_optimized(csv_path)
        
        # Use lighter embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 32}
        )
        
        self.vectorstores: Dict[str, FAISS] = {}
        self._setup_personas()

    def _log_memory_usage(self, stage: str):
        """Log current memory usage"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb

    def _load_csv_optimized(self, csv_path: str) -> pd.DataFrame:
        """Load CSV with memory optimization"""
        try:
            columns_to_load = ['text', 'predicted_persona']
            df = pd.read_csv(
                csv_path,
                usecols=columns_to_load,
                dtype={'text': 'string', 'predicted_persona': 'string'},
                engine='c',
                low_memory=False
            )
            df = self._optimize_dataframe(df)
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return pd.DataFrame()

    def _load_csv_chunked(self, csv_path: str) -> pd.DataFrame:
        """Load CSV in chunks if memory is an issue"""
        chunks = []
        columns_to_load = ['text', 'predicted_persona']
        
        for chunk in pd.read_csv(csv_path, chunksize=self.chunk_size, usecols=columns_to_load):
            chunk = self._optimize_dataframe(chunk)
            chunks.append(chunk)
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 2000:
                break
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
            return df
        else:
            return pd.DataFrame()

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage"""
        if 'text' not in df.columns or 'predicted_persona' not in df.columns:
            raise ValueError(f"Required columns not found. Available: {list(df.columns)}")
        
        df = df[['predicted_persona', 'text']].copy()
        df = df.dropna()
        df['text'] = df['text'].astype('string')
        df['predicted_persona'] = df['predicted_persona'].astype('string')
        df = df[(df['text'].str.len() > 10) & (df['text'].str.len() < 1000)]
        
        persona_counts = df['predicted_persona'].value_counts()
        top_personas = persona_counts.head(self.max_personas).index
        df = df[df['predicted_persona'].isin(top_personas)]
        
        return df

    def _setup_personas(self):
        """Set up persona vectorstores with memory management"""
        if self.df.empty:
            return
            
        personas = self.df['predicted_persona'].unique()[:self.max_personas]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, persona in enumerate(personas):
            try:
                status_text.text(f"Processing persona {i+1}/{len(personas)}: {persona}")
                
                persona_data = self.df[self.df['predicted_persona'] == persona]
                persona_dialogues = persona_data['text'].tolist()[:self.max_rows_per_persona]
                
                if len(persona_dialogues) < 5:
                    continue
                
                unique_dialogues = list(set(persona_dialogues))
                
                if len(unique_dialogues) > self.max_rows_per_persona:
                    unique_dialogues = unique_dialogues[:self.max_rows_per_persona]
                
                batch_size = 100
                if len(unique_dialogues) > batch_size:
                    batches = [unique_dialogues[i:i+batch_size] 
                              for i in range(0, len(unique_dialogues), batch_size)]
                    
                    vectorstore = None
                    for batch in batches:
                        if vectorstore is None:
                            vectorstore = FAISS.from_texts(batch, self.embeddings)
                        else:
                            batch_vs = FAISS.from_texts(batch, self.embeddings)
                            vectorstore.merge_from(batch_vs)
                            del batch_vs
                            gc.collect()
                    
                    self.vectorstores[persona] = vectorstore
                else:
                    self.vectorstores[persona] = FAISS.from_texts(unique_dialogues, self.embeddings)
                
                gc.collect()
                progress_bar.progress((i + 1) / len(personas))
                
            except Exception as e:
                st.error(f"Error processing {persona}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()

    def get_examples(self, persona: str, query: str, k: int = 3) -> str:
        """Retrieve relevant examples for the persona"""
        if persona not in self.vectorstores:
            return f"No examples found for persona: {persona}"

        try:
            docs = self.vectorstores[persona].similarity_search(query, k=k)
            return "\n".join([doc.page_content[:200] + "..." if len(doc.page_content) > 200 
                            else doc.page_content for doc in docs])
        except Exception as e:
            return f"Error retrieving examples for {persona}"

# Helper functions (same as original)
def extract_agent_mention(text: str, available_agents: List[str]) -> Optional[str]:
    """Extract agent mention from text using @ symbol"""
    for agent in available_agents:
        pattern = rf'@{re.escape(agent)}\b'
        if re.search(pattern, text, re.IGNORECASE):
            return agent
    return None

def router_node(state: AgentState):
    """Route to appropriate agent based on @ mentions"""
    last_message = state['messages'][-1].content
    mentioned_agent = extract_agent_mention(last_message, state['available_agents'])

    if mentioned_agent:
        return {
            "messages": state['messages'],
            "current_agent": mentioned_agent,
            "available_agents": state['available_agents']
        }
    else:
        available_list = ", ".join([f"@{agent}" for agent in state['available_agents']])
        response_msg = AIMessage(content=f"Please specify which agent you'd like to talk to. Available: {available_list}")
        return {
            "messages": state['messages'] + [response_msg],
            "current_agent": None,
            "available_agents": state['available_agents']
        }

def create_rag_agent_node(persona: str, persona_retriever: 'PersonaRetriever', llm):
    def agent_func(state: AgentState):
        try:
            last_message = state['messages'][-1].content
            clean_message = re.sub(rf'@{re.escape(persona)}\s*', '', last_message, flags=re.IGNORECASE).strip()

            examples = persona_retriever.get_examples(persona, clean_message, k=2)

            prompt = f"""You are {persona}. Use this style:

{examples[:500]}...

User: {clean_message}

Respond as {persona}:"""

            response = llm.invoke([HumanMessage(content=prompt)])
            
            return {
                "messages": state['messages'] + [response],
                "current_agent": None,
                "available_agents": state['available_agents']
            }
        except Exception as e:
            error_msg = AIMessage(content=f"Sorry, I encountered an error: {str(e)[:100]}")
            return {
                "messages": state['messages'] + [error_msg],
                "current_agent": None,
                "available_agents": state['available_agents']
            }
    
    return agent_func

def should_route_to_agent(state: AgentState) -> str:
    """Determine which node to go to next"""
    return state.get('current_agent', END)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'conv_manager' not in st.session_state:
    st.session_state.conv_manager = None
if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = None

def initialize_system(csv_path: str, openai_key: str):
    """Initialize the system"""
    os.environ["OPENAI_API_KEY"] = openai_key
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    persona_retriever = PersonaRetriever(
        csv_path,
        max_rows_per_persona=1000,
        max_personas=4,
        chunk_size=5000
    )
    
    return llm, persona_retriever

def build_graph(llm, persona_retriever):
    """Build the conversation graph"""
    builder = StateGraph(AgentState)
    personas = list(persona_retriever.vectorstores.keys())
    
    builder.add_node("router", router_node)
    
    for persona in personas:
        builder.add_node(persona, create_rag_agent_node(persona, persona_retriever, llm))
    
    builder.set_entry_point("router")
    builder.add_conditional_edges(
        "router",
        should_route_to_agent,
        {persona: persona for persona in personas} | {END: END}
    )
    
    for persona in personas:
        builder.add_edge(persona, END)
    
    return builder.compile(), personas

class ConversationManager:
    def __init__(self, graph, available_agents):
        self.graph = graph
        self.available_agents = available_agents
        self.conversation_history = []

    def chat(self, user_input: str):
        """Handle a single chat interaction"""
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-6:]
        
        user_msg = HumanMessage(content=user_input)
        self.conversation_history.append(user_msg)

        response = self.graph.invoke({
            "messages": self.conversation_history,
            "current_agent": None,
            "available_agents": self.available_agents
        })

        self.conversation_history = response['messages']
        return response['messages'][-1].content

# Main Streamlit App
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   font-size: 3rem; font-weight: bold; margin: 0;'>
            🤖 Multi-Agent RAG Chat
        </h1>
        <p style='color: #666; font-size: 1.2rem; margin-top: 0.5rem;'>
            Intelligent conversations with persona-based AI agents
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # System parameters
        st.markdown("### 🔧 System Parameters")
        max_personas = st.slider("Max Personas", 2, 10, 4)
        max_rows = st.slider("Max Rows per Persona", 500, 2000, 1000)
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing system..."):
                try:
                    # Use existing CSV file and OpenAI key
                    csv_path = 'cleaned_data.csv'
                    openai_key = "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"
                    
                    # Initialize system
                    llm, persona_retriever = initialize_system(csv_path, openai_key)
                    graph, personas = build_graph(llm, persona_retriever)
                    
                    # Create conversation manager
                    st.session_state.conv_manager = ConversationManager(graph, personas)
                    st.session_state.system_initialized = True
                    st.session_state.personas = personas
                    
                    st.success("✅ System initialized successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Initialization failed: {e}")

        # System status
        if st.session_state.system_initialized:
            st.markdown("### 📊 System Status")
            st.success("🟢 System Ready")
            
            # Available agents
            if 'personas' in st.session_state:
                st.markdown("### 🤖 Available Agents")
                for persona in st.session_state.personas:
                    st.markdown(f'<div class="agent-badge">@{persona}</div>', unsafe_allow_html=True)
            
            # Memory usage
            memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            st.markdown(f"""
            <div class="memory-info">
                <strong>💾 Memory Usage:</strong> {memory_mb:.1f} MB
            </div>
            """, unsafe_allow_html=True)
            
            # Clear conversation
            if st.button("🗑️ Clear Conversation"):
                st.session_state.messages = []
                st.rerun()

    # Main chat interface
    if st.session_state.system_initialized and st.session_state.conv_manager:
        # Agent selection buttons
        st.markdown('<div class="agent-buttons-container">', unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.personas))
        for i, persona in enumerate(st.session_state.personas):
            with cols[i]:
                if st.button(
                    f"🤖 {persona}",
                    key=f"agent_{persona}",
                    help=f"Select {persona} as your conversation partner",
                    type="primary" if st.session_state.selected_agent == persona else "secondary"
                ):
                    st.session_state.selected_agent = persona
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Chat container
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display chat messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ai-message">
                        <strong>AI:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Input form
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "",
                    placeholder=f"Chat with {st.session_state.selected_agent if st.session_state.selected_agent else 'an agent'} (select an agent above)",
                    key="user_input",
                    disabled=not st.session_state.selected_agent
                )
            
            with col2:
                submit_button = st.form_submit_button(
                    "Send 📤",
                    type="primary",
                    disabled=not st.session_state.selected_agent
                )
            
            # Handle form submission
            if submit_button and user_input and st.session_state.selected_agent:
                # Add @ mention to the message
                agent_message = f"@{st.session_state.selected_agent} {user_input}"
                
                # Add user message
                st.session_state.messages.append({"role": "user", "content": agent_message})
                
                # Show loading spinner
                with st.spinner("AI is thinking..."):
                    try:
                        # Get AI response
                        response = st.session_state.conv_manager.chat(agent_message)
                        
                        # Add AI response
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                # Rerun to update chat
                st.rerun()
    
    else:
        # Welcome message
        st.markdown("""
        <div class="system-message">
            <h3>👋 Welcome to Multi-Agent RAG Chat!</h3>
            <p>Please configure the system in the sidebar to get started:</p>
            <ol style="text-align: left; display: inline-block;">
                <li>Upload your CSV dataset</li>
                <li>Enter your OpenAI API key</li>
                <li>Click "Initialize System"</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Features showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h3>🧠 Smart Personas</h3>
                <p>AI agents with distinct personalities trained on your data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h3>🔍 RAG-Powered</h3>
                <p>Retrieval-augmented generation for context-aware responses</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h3>💬 Natural Chat</h3>
                <p>ChatGPT-like interface for seamless conversations</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()