import os
import logging
from typing import List, Optional, Dict, Any
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.agents import Tool, initialize_agent, AgentType
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import requests
import ast
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Application configuration
CONFIG = {
    "document_paths": [
        "nvidia_overview.txt",
        "geforce_now_faq.txt",
        "rtx_50_series_specs.txt",
        "rtx_pro_6000.txt",
        "investor_faq.txt"
    ],
    "chunk_size": 500,
    "chunk_overlap": 50,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_new_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "retriever_k": 3
}

# ---- Document Loading ---- #
@st.cache_resource
def load_documents() -> List[Any]:
    """
    Load and split text documents into chunks for vector store processing.
    Returns a list of document chunks with metadata.
    """
    logger.info("Loading documents...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"]
        )
        all_chunks = []
        for path in CONFIG["document_paths"]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Document not found: {path}")
            loader = TextLoader(path)
            chunks = loader.load_and_split(text_splitter)
            for chunk in chunks:
                chunk.metadata["title"] = os.path.basename(path)
            all_chunks.extend(chunks)
        if not all_chunks:
            raise ValueError("No documents loaded.")
        logger.info(f"Loaded {len(all_chunks)} document chunks.")
        return all_chunks
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        st.error(f"Failed to load documents: {str(e)}")
        return []

# ---- Vector Store Creation ---- #
@st.cache_resource
def create_vector_store(_documents: List[Any]) -> Optional[FAISS]:
    """
    Create a FAISS vector store from document chunks using HuggingFace embeddings.
    Args:
        _documents: List of document chunks.
    Returns:
        FAISS vector store or None if creation fails.
    """
    logger.info("Creating vector store...")
    try:
        if not _documents:
            raise ValueError("No documents provided for vector store.")
        embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])
        vector_store = FAISS.from_documents(_documents, embeddings)
        logger.info("Vector store created successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        st.error(f"Failed to create vector store: {str(e)}")
        return None

# ---- LLM Initialization ---- #
@st.cache_resource
def initialize_llm() -> Optional[HuggingFacePipeline]:
    """
    Initialize the TinyLlama model with 4-bit quantization for low-memory usage.
    Returns:
        HuggingFacePipeline LLM or None if initialization fails.
    """
    logger.info("Initializing LLM...")
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            CONFIG["llm_model"],
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_model"], trust_remote_code=True)
        text_gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=CONFIG["max_new_tokens"],
            do_sample=True,
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            truncation=True
        )
        logger.info("LLM initialized successfully.")
        return HuggingFacePipeline(pipeline=text_gen)
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

# ---- Context Retrieval ---- #
def retrieve_context(query: str, vector_store: FAISS, k: int = CONFIG["retriever_k"]) -> List[str]:
    """
    Retrieve relevant document chunks for a given query using the vector store.
    Args:
        query: User query string.
        vector_store: FAISS vector store.
        k: Number of documents to retrieve.
    Returns:
        List of document content strings.
    """
    logger.info(f"Retrieving context for query: {query}")
    try:
        if not vector_store:
            raise ValueError("Vector store is not initialized.")
        docs = vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(docs)} context chunks.")
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        st.error(f"Failed to retrieve context: {str(e)}")
        return []

# ---- Answer Generation ---- #
def generate_answer(query: str, context_chunks: List[str], llm: HuggingFacePipeline) -> str:
    """
    Generate an answer for the query using the LLM and context chunks.
    Args:
        query: User query string.
        context_chunks: List of relevant document content.
        llm: Initialized LLM pipeline.
    Returns:
        Generated answer string.
    """
    logger.info(f"Generating answer for query: {query}")
    try:
        if not llm:
            raise ValueError("LLM is not initialized.")
        context_str = "\n".join(context_chunks) if context_chunks else "No context available."
        prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        prompt = prompt_template.format(context=context_str, query=query)
        response = llm(prompt)
        answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response
        logger.info("Answer generated successfully.")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        st.error(f"Failed to generate answer: {str(e)}")
        return "Unable to generate answer."

# ---- Tool Definitions ---- #
def calculator_tool(expr: str) -> str:
    """
    Evaluate a mathematical expression.
    Args:
        expr: String containing the mathematical expression.
    Returns:
        Result of the calculation or error message.
    """
    logger.info(f"Evaluating expression: {expr}")
    try:
        result = str(ast.literal_eval(expr))
        logger.info(f"Calculation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Calculation error: {str(e)}")
        return "Error in calculation: Invalid expression."

def dictionary_tool(word: str) -> str:
    """
    Fetch the definition of a word from an online dictionary API.
    Args:
        word: Word to define.
    Returns:
        Definition or error message.
    """
    logger.info(f"Fetching definition for word: {word}")
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=5)
        response.raise_for_status()
        definition = response.json()[0]["meanings"][0]["definitions"][0]["definition"]
        logger.info(f"Definition found: {definition}")
        return definition
    except Exception as e:
        logger.error(f"Dictionary error: {str(e)}")
        return "Definition not found."

# ---- Agent Initialization ---- #
def initialize_agent_with_tools(llm: HuggingFacePipeline, vector_store: FAISS) -> Optional[Any]:
    """
    Initialize a LangChain agent with RAG, calculator, and dictionary tools.
    Args:
        llm: Initialized LLM pipeline.
        vector_store: FAISS vector store.
    Returns:
        Initialized agent or None if initialization fails.
    """
    logger.info("Initializing agent...")
    try:
        if not llm or not vector_store:
            raise ValueError("LLM or vector store not initialized.")
        
        def rag_tool(query: str) -> str:
            context = retrieve_context(query, vector_store)
            return generate_answer(query, context, llm)

        tools = [
            Tool(
                name="Calculator",
                func=calculator_tool,
                description="Use for mathematical calculations (e.g., 'calculate 5 + 3')."
            ),
            Tool(
                name="Dictionary",
                func=dictionary_tool,
                description="Use to find definitions of words (e.g., 'define GPU')."
            ),
            Tool(
                name="RAG",
                func=rag_tool,
                description="Use for general knowledge and contextual answers about NVIDIA."
            )
        ]
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15,
            max_execution_time=30
        )
        logger.info("Agent initialized successfully.")
        return agent
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        st.error(f"Failed to initialize agent: {str(e)}")
        return None

# ---- Streamlit Application ---- #
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="NVIDIA Q&A Assistant", layout="centered")
    st.title("🤖 RAG-Powered NVIDIA Q&A Assistant")
    st.markdown(
        "Ask about NVIDIA products, company, or services, or use 'calculate' or 'define' for specific tasks."
    )

    # Initialize session state
    if "initialized" not in st.session_state:
        with st.spinner("Initializing assistant..."):
            documents = load_documents()
            if not documents:
                st.stop()
            vector_store = create_vector_store(documents)
            if not vector_store:
                st.stop()
            llm = initialize_llm()
            if not llm:
                st.stop()
            agent = initialize_agent_with_tools(llm, vector_store)
            if not agent:
                st.stop()
            st.session_state.update({
                "store": vector_store,
                "llm": llm,
                "agent": agent,
                "documents": documents,
                "initialized": True
            })
            logger.info("Application initialized successfully.")

    # User input
    query = st.text_input(
        "Enter your question (e.g., 'What is GeForce NOW?', 'calculate 5 * 3', 'define GPU'):"
    )
    
    if query:
        if len(query.strip()) < 3:
            st.error("Query is too short. Please provide a more detailed question.")
            return
        with st.spinner("Processing query..."):
            try:
                agent = st.session_state.agent
                vector_store = st.session_state.store
                result = agent.run(query)
                
                # Determine tool used and retrieve context for RAG
                context = []
                tool_used = "Agent (unknown tool)"
                if "RAG" in str(result) or not ("Calculator" in str(result) or "Dictionary" in str(result)):
                    context = retrieve_context(query, vector_store)
                    tool_used = "RAG"
                elif "Calculator" in str(result):
                    tool_used = "Calculator"
                elif "Dictionary" in str(result):
                    tool_used = "Dictionary"

                # Display results
                if context:
                    with st.expander("🔎 Retrieved Context"):
                        for i, chunk in enumerate(context, 1):
                            st.markdown(f"**Snippet {i}:** {chunk}")
                else:
                    st.info("No context snippets available for this query (e.g., calculator or dictionary task).")
                
                st.success(f"💡 Answer: {result}")
                st.markdown(f"**Tool Used:** {tool_used}")
                logger.info(f"Query processed successfully: {query}")
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                st.error(f"Failed to process query: {str(e)}")

if __name__ == "__main__":
    main()