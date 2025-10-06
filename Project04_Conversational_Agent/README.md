# Project 4: Conversational Agent with Task Planning & Sandbox Execution

---

This project is a sophisticated conversational agent built with **Python**, **Streamlit**, and **Google's Gemini API**. 

The agent is designed to go beyond simple conversation by **planning complex tasks**, **generating executable Python code**, and **running that code in a secure, sandboxed environment**.

It provides a user-friendly web interface where users can interact with the agent to perform coding-based tasks (like data analysis or visualization), **without writing any code themselves**.

---

## 🚀 Features

- **💬 Interactive Chat UI**  
  A modern Streamlit interface that supports structured messages, code blocks, and image attachments.

- **🧩 Task Planning (Gemini API)**  
  Automatically decomposes natural language requests (e.g., “create a bar chart of sales”) into structured, executable Python steps.

- **🛡️ Secure Code Execution**  
  Runs AI-generated Python code safely in a **sandboxed subprocess**, with strict timeouts and no filesystem access.

- **📊 Integrated Visualizations**  
  The agent can generate and display **data visualizations** (e.g., Matplotlib charts) directly in the chat.

- **⚡ Command-Driven Interface**  
  Use simple commands to control the agent’s workflow:
  - `/plan` → Generate a plan
  - `/runplan` → Execute the generated plan
  - `/exec` → Run custom Python code

---

## ⚙️ How It Works

The application operates on a powerful **command-driven workflow**:

1. **📝 Planning (`/plan <task>`)**  
   The user provides a natural language task. The backend sends it to the **Gemini API**, which returns a **structured JSON plan** with step-by-step code.

2. **⚙️ Execution (`/runplan`)**  
   The app merges all generated code snippets into a single script for execution.

3. **🔒 Sandboxing**  
   The script is executed in a **secure, isolated subprocess** using Python’s `subprocess` and `tempfile` modules.  
   - This prevents unauthorized file access or infinite loops.

4. **📤 Displaying Results**  
   Output from the sandbox (stdout & stderr) is captured and displayed in a structured format. 
   - If visualizations (like `.png` charts) are generated, they are automatically rendered inline in the Streamlit UI.

---

## 🛠️ Tech Stack

| Component | Technology |
|------------|-------------|
| **Backend** | Python |
| **AI Model** | Google Gemini 1.5 Pro |
| **Frontend** | Streamlit |
| **Visualization** | Matplotlib |
| **Sandboxing** | Python `subprocess` + `tempfile` |

---

## 📋 Setup and Installation

### Prerequisites
- Python **3.8+**
- A **Google API Key** with access to the Gemini API

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your environment variables**
   Create a `.env` file in the root directory:
   ```bash
   GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
   ```

---

## ▶️ Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will launch in your browser automatically.

---

## 💡 Usage

Use the chat input at the bottom of the page to interact with the agent.

### Commands

| Command | Description | Example |
|----------|--------------|----------|
| `/plan <task>` | Ask the agent to create a plan for a task | `/plan create a pie chart with 3 slices for apples, bananas, and cherries` |
| `/runplan` | Execute the most recent plan | `/runplan` |
| `/exec <python code>` | Run custom Python code in the sandbox | `/exec import numpy as np; print(np.random.rand(3))` |

