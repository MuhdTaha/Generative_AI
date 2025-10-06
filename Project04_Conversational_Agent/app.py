import streamlit as st
from agent_tools import handle_special_commands
import json
import re

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="Conversational Task Agent",
    page_icon="🤖",
    layout="wide"
)

# ----------------------
# Title & Description
# ----------------------
st.title("🤖 Conversational Agent with Task Planning & Sandbox Execution")
st.markdown("""
Welcome! This web app provides an intuitive interface for a powerful conversational agent. 
You can interact with it using the following commands in the chat box below:

- **/plan `<task>`**: The agent will break down your task into a series of clear, executable steps.
  - *Example:* `/plan create a bar chart showing sales for Q1, Q2, and Q3`
- **/runplan**: The agent will execute the most recently created plan in a secure sandbox.
  - *Example:* `/runplan`
- **/exec `<python code>`**: The agent will run a specific Python code snippet in the sandbox.
  - *Example:* `/exec print("Hello from the sandbox!")`
""")
st.markdown("---")

# ----------------------
# Session State Initialization
# ----------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "state" not in st.session_state:
    st.session_state.state = {} # Holds the last plan

# ----------------------
# Conversation Display
# ----------------------
# Loop through the conversation history and display each message
for message in st.session_state.conversation:
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        # Handle different types of content for the assistant
        if role == "assistant" and isinstance(content, dict):
            # Display plans in a structured way
            if content.get("type") == "plan":
                st.markdown("Here is the plan I've created:")
                for i, step in enumerate(content["steps"], 1):
                    st.markdown(f"**Step {i}:** {step.get('instruction', 'No instruction.')}")
                    if "code" in step.get("meta", {}):
                        st.code(step["meta"]["code"], language="python")
            
            # Display execution results
            elif content.get("type") == "execution_result":
                st.markdown("I have executed the plan. Here are the results:")
                st.markdown("**Output (stdout):**")
                st.code(f"{content.get('stdout', 'No output.')}", language="bash")
                
                # Conditionally display STDERR only if there is content
                stderr_content = content.get('stderr')
                if stderr_content and stderr_content.strip():
                    st.markdown("**Errors (stderr):**")
                    st.code(f"{stderr_content}", language="bash")

                # Display any images generated during execution
                if "attachments" in content and content["attachments"]:
                    st.markdown("**Visualizations:**")
                    for attachment in content["attachments"]:
                        if attachment.get("type") == "image":
                            st.image(attachment["path"], caption="Generated Visualization")
            
            # Display direct code execution results
            elif content.get("type") == "code_result":
                st.markdown("I have executed the code. Here are the results:")
                st.markdown("**Output (stdout):**")
                st.code(f"{content.get('stdout', 'No output.')}", language="bash")
                
                # Conditionally display STDERR only if there is content
                stderr_content = content.get('stderr')
                if stderr_content and stderr_content.strip():
                    st.markdown("**Errors (stderr):**")
                    st.code(f"{stderr_content}", language="bash")

        # For simple text messages (user and assistant)
        else:
            st.markdown(content)

# ----------------------
# Chat Input Handling
# ----------------------
if prompt := st.chat_input("Enter command or message..."):
    # Add user message to conversation history
    st.session_state.conversation.append({"role": "user", "content": prompt})
    
    # Display the user message immediately
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Agent processes the command
    with st.chat_message("assistant", avatar="🤖"):
        # Show a spinner while the agent is working
        spinner_text = "Thinking..."
        if prompt.startswith("/plan"):
            spinner_text = "Planning task..."
        elif prompt.startswith("/run") or prompt.startswith("/exec"):
            spinner_text = "Executing..."

        with st.spinner(spinner_text):
            # This function contains the core logic for planning and execution
            result = handle_special_commands(prompt, st.session_state.state)
            
            # If the command was handled by the special command logic
            if result["handled"]:
                reply_content = result.get("reply", "An unknown error occurred.")
                attachments = result.get("attachments", [])
                
                # Structure the response for display
                if prompt.startswith("/plan"):
                    try:
                        # Attempt to parse the plan from the raw reply for structured display
                        plan_steps = st.session_state.state.get("last_plan", [])
                        assistant_response = {"type": "plan", "steps": plan_steps}
                    except (json.JSONDecodeError, IndexError):
                        assistant_response = "I couldn't generate a valid plan. Please try again."
                
                elif prompt.startswith("/runplan"):
                    try:
                        # Find the start of the JSON object in the reply string
                        json_start_index = reply_content.find('{')
                        if json_start_index != -1:
                            json_string = reply_content[json_start_index:]
                            # Parse the JSON output from the execution result
                            exec_output = json.loads(json_string)
                            assistant_response = {
                                "type": "execution_result",
                                "stdout": exec_output.get("stdout"),
                                "stderr": exec_output.get("stderr"),
                                "attachments": attachments
                            }
                        else:
                            # If no JSON is found, display the raw reply.
                            assistant_response = reply_content
                    except json.JSONDecodeError:
                         assistant_response = "Execution finished, but the output was not in the expected format."

                elif prompt.startswith("/exec"):
                     # Improved, non-greedy parsing for /exec reply
                    stdout_match = re.search(r"📤 stdout:\n(.*?)\n⚠️ stderr:", reply_content, re.DOTALL)
                    stderr_match = re.search(r"⚠️ stderr:\n(.*)", reply_content, re.DOTALL)
                    assistant_response = {
                        "type": "code_result",
                        "stdout": stdout_match.group(1).strip() if stdout_match else "N/A",
                        "stderr": stderr_match.group(1).strip() if stderr_match else "", # Default to empty
                    }

                else:
                    assistant_response = reply_content
            
            # If it's not a special command, treat it as a general chat message (future extension)
            else:
                assistant_response = "I can only handle `/plan`, `/runplan`, and `/exec` commands right now."

            # Display the structured response
            if isinstance(assistant_response, dict):
                if assistant_response.get("type") == "plan":
                    st.markdown("Here is the plan I've created:")
                    for i, step in enumerate(assistant_response["steps"], 1):
                        st.markdown(f"**Step {i}:** {step.get('instruction', 'No instruction.')}")
                        if "code" in step.get("meta", {}):
                           st.code(step["meta"]["code"], language="python")
                else: # For execution results
                    st.markdown("Execution complete. Here are the results:")
                    st.markdown("**Output (stdout):**")
                    st.code(f"{assistant_response.get('stdout', '')}", language="bash")

                    stderr_content = assistant_response.get('stderr')
                    if stderr_content and stderr_content.strip():
                        st.markdown("**Errors (stderr):**")
                        st.code(f"{stderr_content}", language="bash")

                    if "attachments" in assistant_response and assistant_response["attachments"]:
                        st.markdown("**Visualizations:**")
                        for attachment in assistant_response["attachments"]:
                            if attachment.get("type") == "image":
                                st.image(attachment["path"], caption="Generated Visualization")
            else:
                 st.markdown(assistant_response)

            # Add the final, structured response to the conversation history
            st.session_state.conversation.append({"role": "assistant", "content": assistant_response})

# ----------------------
# Footer
# ----------------------
st.markdown("---")
footer_html = """
<div style="text-align: center; padding: 10px;">
    <p>Developed by <strong>Muhammad Taha</strong></p>
    <a href="https://www.linkedin.com/in/muhdtaha/" target="_blank" style="margin: 0 10px;">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn" width="30" height="30">
    </a>
    <a href="https://github.com/MuhdTaha" target="_blank" style="margin: 0 10px;">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="GitHub" width="30" height="30">
    </a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

