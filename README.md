# NVIDIA Q&A Assistant

## Overview
The NVIDIA Q&A Assistant is a Retrieval-Augmented Generation (RAG) powered web application built with Streamlit and LangChain. It enables users to query information about NVIDIA's products, services, and company details using a  The application leverages a lightweight TinyLlama-1.1B language model, a FAISS vector store for document retrieval, and a LangChain agent to handle three types of queries:
- **General Knowledge Queries**: Contextual answers about NVIDIA based on a predefined knowledge base.
- **Mathematical Calculations**: Evaluation of mathematical expressions.
- **Word Definitions**: Retrieval of word definitions via an external API.

This README gives a clear description of the application architecture, principal design decisions, and how to run the code.

## Architecture
The application is based on a modular design with separate components combined into a web interface using Streamlit. The overview of the major components is presented below:

1. **Document Loader**:
   - **Purpose**: Preprocesses and loads text files with NVIDIA information.
   - **Implementation**: Reads `.txt` files using `langchain_community.document_loaders.TextLoader` and breaks documents into segments (500 characters with 50-character overlap) using `RecursiveCharacterTextSplitter`.
   - **Output**: A list of document chunks with metadata (e.g., file name as title).

2. **Vector Store**:
   - **Purpose**: Stores document embeddings for efficient similarity-based retrieval.
   - **Implementation**: Employs `FAISS` (Facebook AI Similarity Search) with `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`) to create a vector store.
   - **Process**: Document chunks are embedded into a vector space, enabling fast retrieval of the top-k relevant chunks for a given query.

3. **Language Model (LLM)**:
   - **Purpose**: Generates human-like responses based on retrieved context and user queries.
   - **Implementation**: Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with 4-bit quantization (`BitsAndBytesConfig`) to fit within 4GB VRAM constraints. The model is integrated via `HuggingFacePipeline` for text generation.
   - **Configuration**: Limited to 200 new tokens with sampling (temperature=0.7, top_p=0.9) to balance creativity and coherence.

4. **RAG Pipeline**:
   - **Purpose**: Merges document retrieval and LLM generation to return contextual answers.
   - **Implementation**: Retrieves top-3 document chunks with FAISS, templates them with a `PromptTemplate`, and submits the prompt to the LLM to generate answers.

5. **Tools**:
   - **Calculator Tool**: Evaluates mathematical expressions using `ast.literal_eval` for safe parsing.
   - **Dictionary Tool**: Queries the Dictionary API (`api.dictionaryapi.dev`) to fetch word definitions.
   - **RAG Tool**: Wraps the RAG pipeline for agent-based query handling.

6. **Agent**:
   - **Purpose**: Intelligently selects the appropriate tool (RAG, Calculator, or Dictionary) based on the query.
   - **Implementation**: Uses `langchain.agents` with `AgentType.ZERO_SHOT_REACT_DESCRIPTION` to reason about tool selection. Configured with a maximum of 15 iterations and a 30-second timeout.

7. **Streamlit UI**:
   - **Purpose**: Provides a user-friendly interface for input queries and shows results.
   - **Implementation**: Built with Streamlit, including text input, context expander for RAG queries, and tool usage indicator. Caching (`@st.cache_resource`) is utilized to speed up initialization.

8. **Logging**:
   - **Purpose**: Traces application flow and faults for debugging.
   - **Implementation**: Utilizes Python's `logging` module with timestamped INFO and ERROR logs.

## Key Design Choices
The application design achieves maximum efficiency, modularity, and usability with the following key decisions:

1. **Lightweight Model**:
   - **Choice**: TinyLlama-1.1B with 4-bit quantization.
   - **Rationale**:  It is optimal on 4GB VRAM, thus is largely available on consumer-level GPUs or CPUs. Context window of 2048 tokens is enough for short documents and queries.
   - **Trade-off**: Limited in reasoning capacity compared to bigger models, but this is alleviated through rigorous document chunking and prompt instruction clarity.

2. **RAG Approach**:
   - **Choice**: Merges retrieval with FAISS and generation with LLM.
   - **Rationale**: Get a response based on a filtered knowledge, thus increasing relevance while limiting the possibilities of hallucination. FAISS is light and efficient for compact sets of documents.
   - **Trade-off**: Knowledge is limited to the input text files and must be updated by hand to grow.

3. **Agent-Based Tool Selection**:
   - **Choice**: LangChain agent with three tools (RAG, Calculator, Dictionary).
   - **Rationale**: Enables flexible query handling (e.g., math, definitions, or general knowledge) without complex rule-based logic. The zero-shot ReAct agent reasons effectively with minimal configuration.
   - **Trade-off**: Occasional tool misselection, mitigated by verbose logging and error handling.

4. **Streamlit for UI**:
   - **Choice**: Streamlit for quick web application development.
   - **Rationale**: Facilitating development of user interfaces with Python with focus laid on backend logic, performance tuning, and reuse of initialized objects through caching.
   - **Trade-off**: Less specialized than tools like Flask or React but still appropriate for question-and-answer interface.

5. **Configuration Management**:
   - **Choice**:  CONFIG dictionary of parameters (chunk size, model name, etc.) in one place.
   - **Rationale**: It becomes very easy to tune and modify without requiring any change in the code. Parameters are also nicely documented so that they are easy to comprehend.
   - **Trade-off**: Static configuration requires code edits for dynamic changes, acceptable for this scope.

6. **Error Handling and Logging**:
   - **Choice**: Strong try-except blocks then logging.
   - **Rationale**: For strong operation and good error message, while logs help debugging during development and during deployment.
   - **Trade-off**: Complexity in code raised and warranted due to enhancement of reliability.

## Prerequisites
- **Python**: Version 3.8 or higher.
- **Hardware**: GPU with at least 4GB VRAM (for quantized model) or CPU with 8GB+ RAM.
- **Dependencies**: Listed in `requirements.txt` (see Installation).
- **Document Files**: The following text files must be present in the project directory:
  - `nvidia_overview.txt`
  - `geforce_now_faq.txt`
  - `rtx_50_series_specs.txt`
  - `rtx_pro_6000.txt`
  - `investor_faq.txt`

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd nvidia-qa-assistant
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Content `requirements.txt`:
   ```
   streamlit
   langchain
   langchain-community
   transformers
   torch
   faiss-cpu
   sentence-transformers
   requests
   bitsandbytes
   ```

4. **Verify Document Files**:
   - Ensure the listed `.txt` files are in the project directory. These files form the knowledge base for RAG queries.

## Running the Code
1. **Start the Application**:
   ```bash
   streamlit run tinny_llama.py
   ```
   This command launches the Streamlit server, typically on `http://localhost:8501`.

2. **Access the Web Interface**:
   - Open a web browser and navigate to `http://localhost:8501`.
   - The interface displays a title, description, and text input field.

3. **Interact with the Assistant**:
   - Enter a query, such as:
     - "What is GeForce NOW?"
     - "calculate 5 * 3"
     - "define GPU"
   - The application processes the query and displays:
     - The answer.
     - The tool used (RAG, Calculator, or Dictionary).
     - Retrieved context snippets (for RAG queries) in an expandable section.

4. **Monitor Logs**:
   - Check the terminal for logs indicating initialization, query processing, and any errors.

## Troubleshooting
- **Missing Documents**: Verify that all `.txt` files are in the project directory and match `CONFIG["document_paths"]`.
- **Model Loading Issues**: Ensure sufficient VRAM (4GB+) or switch to CPU mode (slower). Check `torch` and `bitsandbytes` installations.
- **Dependency Errors**: Re-run `pip install -r requirements.txt` or update packages (`pip install --upgrade <package>`).
- **Query Failures**: Make sure queries are concise and a minimum of 3 characters. Review logs for certain errors.
- **API Issues**: The internet is needed for the dictionary tool. Check connectivity if definitions don't work.
## Limitations
- **Model Constraints**: TinyLlama-1.1B can be afflicted with sophisticated reasoning because of its size and 2048-token limit.
- **Knowledge Base**: Confined to the given `.txt` files. Expansion of the base is accomplished by importing additional files and modification of `CONFIG`.
- **Internet Dependency**: Dictionary tool depends on an active internet connection.
- **Scalability**: Document sets of small size and user interaction by a single user.

## Future Improvements
- **Larger Model**: Integrate a more powerful LLM (e.g., Llama-7B) for improved reasoning, if hardware allows.
- **Dynamic Knowledge Base**: Support uploading new documents via the UI.
- **Advanced Retrieval**: Implement hybrid search (e.g., BM25 + embeddings) for better context relevance.
- **Caching Enhancements**: Persist vector store to disk to reduce initialization time.
  
## Author
👤 Shubham Sontakke  
🔗 GitHub: https://github.com/Shubhamgs81
