import os
import streamlit as st
from dotenv import load_dotenv
import time
import math
from typing import List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import google.generativeai as genai

# Attempt to import necessary libraries
try:
    import pytesseract
    from pdf2image import convert_from_path, exceptions as pdf2image_exceptions
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langgraph.graph import END, StateGraph
except ImportError as e:
    st.error(f"A required library is not installed. Please run 'pip install -r requirements.txt'. Error: {e}")
    st.stop()

# --- App Configuration & Secrets ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
POPPLER_PATH = os.getenv("POPPLER_PATH")

# --- Helper Functions ---
def process_page_ocr(page, image):
    """Run OCR on a page if it has no text content."""
    if not page.page_content.strip():
        try:
            page.page_content = pytesseract.image_to_string(image)
        except pytesseract.TesseractNotFoundError:
            st.error("Tesseract is not installed or not in your PATH. Please install Tesseract and try again.")
            st.stop()
    return page

def load_pdf_with_ocr(pdf_path: str, poppler_path: str, num_threads: int = 8, dpi: int = 150):
    """Load PDF and ensure all pages have text (OCR fallback)."""
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found at path: {pdf_path}")
        return None
        
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_page_ocr, page, images[idx]) for idx, page in enumerate(docs)]
            processed_pages = [f.result() for f in as_completed(futures) if f.result() is not None]

        return processed_pages
    except pdf2image_exceptions.PDFInfoNotInstalledError:
        st.error(
            "Poppler Error: The application could not find the Poppler installation. "
            "Please ensure Poppler is installed and that the `POPPLER_PATH` in your `.env` file "
            "points to the correct 'bin' directory."
        )
        return None
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        return None

def format_text_for_summarization(processed_pages) -> str:
    """Join extracted text from pages into a single string."""
    return "\n\n".join(page.page_content for page in processed_pages if page.page_content)

def chunk_text(text: str, chunk_size: int):
    """Split text into chunks of roughly chunk_size words."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# --- Gemini LLM Call ---
def call_gemini(prompt: str, model_name: str = "gemini-1.5-flash-latest") -> str:
    """
    Calls the Google Gemini API to generate content based on a prompt.
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        # Clean up the response text, removing markdown formatting
        if response.parts:
            return response.text.strip().replace("```cypher", "").replace("```json", "").replace("```", "").strip()
        else:
            print("Warning: Gemini API returned an empty response.")
            return ""
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {str(e)}")
        return ""

# --- LangChain & LangGraph Core Logic ---
def create_prompts():
    """Create the prompt templates for generation and reflection."""
    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in document analysis. Generate the best possible summary for the provided text. "
                   "If the user provides a critique of your summary, you must respond with a revised, improved version."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert writing critic. Your task is to critique the provided summary. "
                   "Identify flaws, missing information, or areas for improvement. Provide specific, actionable recommendations."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    return generation_prompt, reflection_prompt

class GraphState(TypedDict):
    messages: List[BaseMessage]
    loop_count: int

def build_graph(generation_prompt, reflection_prompt, max_loops=3):
    """Build the reflection loop graph."""
    
    def generation_node(state: GraphState):
        prompt_str = generation_prompt.invoke({"messages": state["messages"]}).to_string()
        result = call_gemini(prompt_str)
        return {"messages": state["messages"] + [AIMessage(content=result)], "loop_count": state.get("loop_count", 0)}

    def reflection_node(state: GraphState):
        prompt_str = reflection_prompt.invoke({"messages": state["messages"]}).to_string()
        critique = call_gemini(prompt_str)
        return {"messages": state["messages"] + [HumanMessage(content=critique)], "loop_count": state.get("loop_count", 0) + 1}

    def should_continue(state: GraphState):
        return "reflection" if state.get("loop_count", 0) < max_loops else END

    builder = StateGraph(GraphState)
    builder.add_node("generation", generation_node)
    builder.add_node("reflection", reflection_node)
    builder.set_entry_point("generation")
    builder.add_conditional_edges("generation", should_continue)
    builder.add_edge("reflection", "generation")
    
    return builder.compile()

def run_batch_summaries(processed_pages, graph, batch_size, chunk_words):
    """Run the summarization + reflection loop in token-safe batches."""
    response_list = []
    log_list = []
    num_batches = math.ceil(len(processed_pages) / batch_size)

    progress_bar = st.progress(0, text="Summarizing batches...")
    
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, len(processed_pages))
        batch_text = format_text_for_summarization(processed_pages[start_index:end_index])
        
        if not batch_text.strip():
            continue

        for chunk in chunk_text(batch_text, chunk_size=chunk_words):
            inputs = {"messages": [HumanMessage(content=f"Summarize this text:\n\n{chunk}")], "loop_count": 0}
            response = graph.invoke(inputs)
            response_list.append(response['messages'][-1])
            log_list.append(response['messages'])

        progress_bar.progress((i + 1) / num_batches, text=f"Summarizing batch {i+1}/{num_batches}")
        
    progress_bar.empty()
    return response_list, log_list

def combine_and_summarize(response_list, chunk_words):
    """Combine all batch summaries and run a final summarization."""
    summaries = [response.content for response in response_list]
    final_combined_text = "\n" + "-"*50 + "\n".join(summaries)
    
    combined_summary = ""
    for chunk in chunk_text(final_combined_text, chunk_size=chunk_words):
        prompt_template = ChatPromptTemplate.from_template("You are an expert summary writer. Combine and summarize the following summaries into a single, cohesive final summary.\n\nText: {text}\n\nFinal Summary:")
        prompt_str = prompt_template.invoke({"text": chunk}).to_string()
        chunk_summary = call_gemini(prompt_str)
        combined_summary += chunk_summary + "\n"
        
    return combined_summary

# --- Streamlit UI ---
st.set_page_config(page_title="Reflection-Critic PDF Summarization", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    /* This targets the sidebar container */
    .css-1d391kg {
        display: flex;
        flex-direction: column;
    }
    /* This is the footer class */
    .sidebar-footer {
        margin-top: auto;
        padding-top: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


st.title("🧠 Reflection-Critic Agents for Efficient PDF Summarization")
st.markdown("This app uses a multi-agent system (LangGraph) to summarize PDF documents. A 'Summarizer' agent creates a summary, and a 'Critic' agent provides feedback for revision, improving the quality of the final output.")

# Initialize session state
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'logs' not in st.session_state:
    st.session_state.logs = None

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    # Check for secrets and provide guidance
    if not GOOGLE_API_KEY:
        st.error("`GOOGLE_API_KEY` not found. Please set it in your `.env` file.")
    if not POPPLER_PATH:
        st.error("`POPPLER_PATH` not found. Please set it in your `.env` file.")

    st.markdown("---")
    st.subheader("Advanced Settings")
    max_loops = st.slider("Reflection Loops", 1, 5, 2, help="Number of times the summary is critiqued and revised.")
    batch_size = st.slider("Pages per Batch", 1, 20, 8, help="Number of PDF pages to process in each batch.")
    chunk_words = st.slider("Words per Chunk", 500, 2000, 1000, help="Max words per LLM call to prevent context overflow.")

    summarize_button = st.button("Summarize PDF", type="primary", use_container_width=True)
    
    # Footer
    footer_html = """
    <div class="sidebar-footer">
        <hr>
        <p>Developed by <strong>Muhammad Taha</strong></p>
        <a href="https://www.linkedin.com/in/muhdtaha/" target="_blank" style="margin: 0 10px;">
            <img src="https://cdn1.iconfinder.com/data/icons/logotypes/32/circle-linkedin-512.png" alt="LinkedIn" width="50" height="50">
        </a>
        <a href="https://github.com/MuhdTaha" target="_blank" style="margin: 0 10px;">
            <img src="https://img.icons8.com/m_sharp/512/FFFFFF/github.png" alt="GitHub" width="50" height="50">
        </a>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# --- Main App Logic ---
if summarize_button:
    # Check for all prerequisites before proceeding
    if not uploaded_file:
        st.warning("Please upload a PDF file.")
    elif not GOOGLE_API_KEY or not POPPLER_PATH:
        st.warning("Please ensure both `GOOGLE_API_KEY` and `POPPLER_PATH` are set in your `.env` file.")
    else:
        # Configure the genai library with the key from .env
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
        except Exception as e:
            st.error(f"Failed to configure Google API key: {e}")
            st.stop()

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        with st.status("Processing PDF... Please wait.", expanded=True) as status:
            try:
                # Step 1: Load PDF with OCR, using POPPLER_PATH from .env
                status.update(label="Step 1/4: Loading PDF and performing OCR...")
                processed_pages = load_pdf_with_ocr(pdf_path, POPPLER_PATH)

                if processed_pages:
                    # Step 2: Build Prompts and Graph
                    status.update(label="Step 2/4: Initializing AI agents and graph...")
                    generation_prompt, reflection_prompt = create_prompts()
                    graph = build_graph(generation_prompt, reflection_prompt, max_loops=max_loops)
                    
                    # Step 3: Run Batch Summaries
                    status.update(label="Step 3/4: Summarizing in batches with reflection...")
                    time.sleep(1) # Give time for UI to update
                    response_list, log_list = run_batch_summaries(processed_pages, graph, batch_size, chunk_words)
                    st.session_state.logs = log_list
                    
                    # Step 4: Combine and Finalize
                    status.update(label="Step 4/4: Generating final cohesive summary...")
                    final_summary = combine_and_summarize(response_list, chunk_words)
                    st.session_state.summary = final_summary
                    
                    status.update(label="Summarization complete!", state="complete", expanded=False)
                else:
                    # Specific Poppler error is handled inside load_pdf_with_ocr, 
                    # so we only need a general failure message here if it returns None.
                    status.update(label="Processing Failed. See error message above.", state="error")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                status.update(label="An error occurred", state="error")
            finally:
                # Clean up the temporary file
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

# --- Display Results ---
if st.session_state.summary:
    st.subheader("Final Summary")
    st.markdown(st.session_state.summary)

if st.session_state.logs:
    with st.expander("View Detailed Reflection Log"):
        for i, log in enumerate(st.session_state.logs):
            st.markdown(f"--- **Chunk {i+1}** ---")
            for j, message in enumerate(log):
                if isinstance(message, HumanMessage):
                    if j == 0:
                        st.info(f"**Initial Text Chunk:**\n\n{message.content[:500]}...")
                    else:
                        st.warning(f"**Critique:**\n\n{message.content}")
                elif isinstance(message, AIMessage):
                    if j == 1:
                        st.success(f"**Initial Summary:**\n\n{message.content}")
                    else:
                         st.success(f"**Revised Summary:**\n\n{message.content}")

