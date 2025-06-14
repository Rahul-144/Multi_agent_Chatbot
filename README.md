# Multi-Agent RAG Chat System

A conversational AI system that uses multiple persona-based agents powered by retrieval-augmented generation (RAG). The system allows users to interact with different AI personas, each trained on specific dialogue patterns.

## Features

- 🤖 Multiple AI personas with distinct conversational styles
- 🔍 RAG-powered responses using FAISS vector store
- 💬 Natural chat interface with @ mentions for agent selection
- 🧠 Memory-optimized processing of large dialogue datasets
- 🚀 Built with LangChain and Streamlit

## Dataset

This project uses the Cornell Movie-Dialogs Corpus dataset:

### Dataset Reference

```bibtex
@InProceedings{Danescu-Niculescu-Mizil+Lee:11a,
  author={Cristian Danescu-Niculescu-Mizil and Lillian Lee},
  title={Chameleons in imagined conversations:
         A new approach to understanding coordination of linguistic style in dialogs.},
  booktitle={Proceedings of the
             Workshop on Cognitive Modeling and Computational Linguistics, ACL 2011},
  year={2011}
}
```

## Data Preprocessing

The raw movie dialogue dataset undergoes several preprocessing steps to prepare it for the multi-agent chat system:

1. **Text Cleaning**
   - Removal of special characters and formatting
   - Standardization of punctuation and spacing
   - Basic spell checking and correction

2. **Dialogue Extraction**
   - Parsing of character-dialogue pairs
   - Conversation thread reconstruction
   - Speaker identification and mapping

3. **Feature Engineering**
   - Sentiment analysis of dialogues
   - Character personality trait extraction
   - Conversation context labeling

4. **Data Formatting**
   - Conversion to structured format for RAG
   - Creation of FAISS-compatible embeddings
   - Generation of training examples for each persona

[Processed Dataset Link - Coming Soon]

## Requirements

- Python 3.8+
- OpenAI API key
- Required packages (install via `pip install -r requirements.txt`):
  - langchain
  - streamlit
  - pandas
  - faiss-cpu
  - sentence-transformers
  - python-dotenv

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Place your processed dataset CSV file in the project directory
5. Run the application:
   - For CLI version: `python main.py`
   - For web interface: `streamlit run app.py`

## Usage

- Start the application and wait for system initialization
- Chat with different personas using @ mentions (e.g., "@persona_name Hello!")
- The system will automatically route your message to the appropriate agent
- Each agent's response is enhanced with relevant examples from the training data

## Memory Optimization

The system includes several memory optimization features:
- Batch processing of large datasets
- Efficient vector store management
- Conversation history pruning
- Resource monitoring and automatic cleanup

## License

MIT License 