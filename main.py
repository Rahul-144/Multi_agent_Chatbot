from typing import List, TypedDict, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
import pandas as pd
import os
import re
import gc
import psutil
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define state
class AgentState(TypedDict):
    messages: List[BaseMessage]
    current_agent: Optional[str]
    available_agents: List[str]

# Memory-optimized persona loader with RAG setup
class PersonaRetriever:
    def __init__(self, csv_path: str, max_rows_per_persona: int = 1000, 
                 max_personas: int = 10, chunk_size: int = 10000):
        self.max_rows_per_persona = max_rows_per_persona
        self.max_personas = max_personas
        self.chunk_size = chunk_size
        
        # Monitor memory usage
        self._log_memory_usage("Initial")
        
        # Load CSV with memory optimization
        self.df = self._load_csv_optimized(csv_path)
        self._log_memory_usage("After CSV load")
        
        # Use lighter embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Force CPU to save GPU memory
            encode_kwargs={'batch_size': 32}  # Smaller batch size
        )
        
        self.vectorstores: Dict[str, FAISS] = {}
        self._setup_personas()
        self._log_memory_usage("After setup complete")

    def _log_memory_usage(self, stage: str):
        """Log current memory usage"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage at {stage}: {memory_mb:.1f} MB")

    def _load_csv_optimized(self, csv_path: str) -> pd.DataFrame:
        """Load CSV with memory optimization - only text and predicted_persona columns"""
        print(f"Loading CSV from: {csv_path}")
        
        try:
            # First, peek at the file to understand its structure
            sample_df = pd.read_csv(csv_path, nrows=5)
            print(f"CSV columns: {list(sample_df.columns)}")
            
            # Only load the required columns to save memory
            columns_to_load = ['text', 'predicted_persona']
            
            # Load with optimizations - only specific columns
            df = pd.read_csv(
                csv_path,
                usecols=columns_to_load,  # Only load required columns
                dtype={'text': 'string', 'predicted_persona': 'string'},  # Use string dtype
                engine='c',  # Use C engine for speed
                low_memory=False
            )
            
            print(f"Loaded {len(df)} total rows with columns: {list(df.columns)}")
            
            # Immediate memory optimization
            df = self._optimize_dataframe(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return self._load_csv_chunked(csv_path)

    def _load_csv_chunked(self, csv_path: str) -> pd.DataFrame:
        """Load CSV in chunks if memory is an issue - only required columns"""
        print("Loading CSV in chunks...")
        chunks = []
        
        # Only load required columns
        columns_to_load = ['text', 'predicted_persona']
        
        for chunk in pd.read_csv(csv_path, chunksize=self.chunk_size, usecols=columns_to_load):
            # Process each chunk
            chunk = self._optimize_dataframe(chunk)
            chunks.append(chunk)
            
            # Monitor memory and break if getting too large
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > 2000:  # Stop if using more than 2GB
                print(f"Memory limit reached, stopping at {len(chunks)} chunks")
                break
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            del chunks  # Free memory
            gc.collect()
            return df
        else:
            raise Exception("Could not load any data from CSV")

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe memory usage - ensure only text and predicted_persona columns"""
        
        # Ensure we have the required columns
        if 'text' not in df.columns or 'predicted_persona' not in df.columns:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Required columns 'text' and 'predicted_persona' not found. Available: {list(df.columns)}")
        
        # Keep only the required columns
        df = df[['predicted_persona', 'text']].copy()
        
        # Clean data
        df = df.dropna()
        df['text'] = df['text'].astype('string')
        df['predicted_persona'] = df['predicted_persona'].astype('string')
        
        # Remove very short or very long texts
        df = df[(df['text'].str.len() > 10) & (df['text'].str.len() < 1000)]
        
        # Limit personas to prevent memory issues
        persona_counts = df['predicted_persona'].value_counts()
        top_personas = persona_counts.head(self.max_personas).index
        df = df[df['predicted_persona'].isin(top_personas)]
        
        print(f"Optimized to {len(df)} rows with {len(top_personas)} personas")
        print(f"Final columns: {list(df.columns)}")
        return df

    def _setup_personas(self):
        """Set up persona vectorstores with memory management"""
        print("Setting up personas with memory optimization...")
        
        personas = self.df['predicted_persona'].unique()[:self.max_personas]
        print(f"Processing {len(personas)} personas: {list(personas)}")

        for i, persona in enumerate(personas):
            try:
                print(f"Processing persona {i+1}/{len(personas)}: {persona}")
                
                # Get dialogues for this persona (limited)
                persona_data = self.df[self.df['predicted_persona'] == persona]
                persona_dialogues = persona_data['text'].tolist()[:self.max_rows_per_persona]
                
                if len(persona_dialogues) < 5:  # Skip personas with too few examples
                    print(f"Skipping {persona} - only {len(persona_dialogues)} examples")
                    continue
                
                # Remove duplicates and very similar texts
                unique_dialogues = list(set(persona_dialogues))
                
                # Limit to prevent memory issues
                if len(unique_dialogues) > self.max_rows_per_persona:
                    unique_dialogues = unique_dialogues[:self.max_rows_per_persona]
                
                print(f"Creating vectorstore for {persona} with {len(unique_dialogues)} dialogues")
                
                # Create vectorstore in batches to manage memory
                batch_size = 100
                if len(unique_dialogues) > batch_size:
                    # Process in batches
                    batches = [unique_dialogues[i:i+batch_size] 
                              for i in range(0, len(unique_dialogues), batch_size)]
                    
                    vectorstore = None
                    for batch_idx, batch in enumerate(batches):
                        if vectorstore is None:
                            vectorstore = FAISS.from_texts(batch, self.embeddings)
                        else:
                            batch_vs = FAISS.from_texts(batch, self.embeddings)
                            vectorstore.merge_from(batch_vs)
                            del batch_vs
                            gc.collect()
                        
                        print(f"  Processed batch {batch_idx+1}/{len(batches)}")
                    
                    self.vectorstores[persona] = vectorstore
                else:
                    self.vectorstores[persona] = FAISS.from_texts(unique_dialogues, self.embeddings)
                
                # Force garbage collection after each persona
                gc.collect()
                self._log_memory_usage(f"After {persona}")
                
            except Exception as e:
                print(f"Error processing {persona}: {e}")
                continue

        print(f"Successfully created vectorstores for {len(self.vectorstores)} personas")

    def get_examples(self, persona: str, query: str, k: int = 3) -> str:
        """Retrieve relevant examples for the persona"""
        if persona not in self.vectorstores:
            return f"No examples found for persona: {persona}"

        try:
            docs = self.vectorstores[persona].similarity_search(query, k=k)
            return "\n".join([doc.page_content[:200] + "..." if len(doc.page_content) > 200 
                            else doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error retrieving examples for {persona}: {e}")
            return f"Error retrieving examples for {persona}"

# Initialize components with memory management
def initialize_system(csv_path: str):
    """Initialize the system with proper memory management"""
    print("Initializing system...")
    
    # Set OpenAI API key (replace with your key)
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = ""
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # Use cheaper model
    
    # Initialize with memory limits
    persona_retriever = PersonaRetriever(
        csv_path,
        max_rows_per_persona=1000,  # Limit examples per persona
        max_personas=4,  # Limit number of personas
        chunk_size=5000  # Smaller chunks
    )
    
    return llm, persona_retriever

def extract_agent_mention(text: str, available_agents: List[str]) -> Optional[str]:
    """Extract agent mention from text using @ symbol"""
    for agent in available_agents:
        pattern = rf'@{re.escape(agent)}\b'
        if re.search(pattern, text, re.IGNORECASE):
            return agent
    return None

def format_messages(messages: List[BaseMessage]) -> str:
    """Format messages with length limits"""
    formatted = []
    for m in messages[-3:]:  # Only last 3 messages
        content = m.content[:300] + "..." if len(m.content) > 300 else m.content
        speaker = 'User' if isinstance(m, HumanMessage) else 'AI'
        formatted.append(f"{speaker}: {content}")
    return "\n".join(formatted)

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

            # Get relevant examples
            examples = persona_retriever.get_examples(persona, clean_message, k=2)  # Fewer examples

            # Shorter prompt to save tokens
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

class ConversationManager:
    def __init__(self, graph, available_agents):
        self.graph = graph
        self.available_agents = available_agents
        self.conversation_history = []

    def chat(self, user_input: str):
        """Handle a single chat interaction with memory management"""
        # Limit conversation history to prevent memory bloat
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-6:]  # Keep last 6 messages
        
        user_msg = HumanMessage(content=user_input)
        self.conversation_history.append(user_msg)

        response = self.graph.invoke({
            "messages": self.conversation_history,
            "current_agent": None,
            "available_agents": self.available_agents
        })

        self.conversation_history = response['messages']
        return response['messages'][-1].content

def main():
    """Main function with error handling"""
    csv_path = '/content/cleaned_data.csv'  # Update this path
    
    try:
        # Initialize system components
        llm, persona_retriever = initialize_system(csv_path)
        
        # Build the graph
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
        
        graph = builder.compile()
        
        # Initialize conversation manager
        conv_manager = ConversationManager(graph, personas)
        
        print("Multi-Agent RAG Chat System (Memory Optimized)")
        print(f"Available agents: {', '.join([f'@{agent}' for agent in personas])}")
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    break
                
                response = conv_manager.chat(user_input)
                print(f"AI: {response}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}\n")
                
    except Exception as e:
        print(f"System initialization failed: {e}")
        print("Try reducing max_personas and max_rows_per_persona in PersonaRetriever")

if __name__ == "__main__":
    main()