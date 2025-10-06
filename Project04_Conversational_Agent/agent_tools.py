import os
from dotenv import load_dotenv
import sys
import json
import tempfile
import textwrap
import subprocess
from typing import List, Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import google.generativeai as genai
import re

# ---------------------------
# 0) Configuration
# ---------------------------
# Define the model name once to be reused throughout the script
DEFAULT_MODEL_NAME = "gemini-2.5-flash"

# ---------------------------
# 1) Task decomposition (Planner)
# ---------------------------
def decompose_task_via_gemini(task, model_name=DEFAULT_MODEL_NAME):
    """
    Uses the specified Gemini model to break down a task into a JSON plan.
    """
    import google.generativeai as genai
    import os
    import re
    import json

    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(model_name)

    prompt = f"""
    You are a task planner agent. Break down the following task into clear, numbered Python code steps.
    Return only a valid JSON object in this format:
    {{
      "steps": [
        {{
          "id": 1,
          "type": "code",
          "instruction": "Describe the step",
          "meta": {{"code": "Python code to run"}}
        }}
      ]
    }}

    Task: "{task}"
    """

    response = model.generate_content(prompt)
    text = response.text.strip()

    # Extract JSON safely
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        print("⚠️ Could not extract JSON from Gemini response.")
        print(text)
        return None

    json_str = match.group(0)
    try:
        plan_obj = json.loads(json_str)
        steps = plan_obj.get("steps", [])
        # Ensure type & meta are present
        for i, step in enumerate(steps):
            step.setdefault("type", "code")
            step.setdefault("meta", {"code": step.get("instruction", "")})
        return steps
    except Exception as e:
        print(f"⚠️ Could not parse JSON response from Gemini:\n{text}")
        print("Error:", e)
        return None

# ---------------------------
# 2) Code sandbox execution (with safe imports)
# ---------------------------
def run_code_sandbox(code_str: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Execute Python code in a sandbox subprocess.
    Returns a dict: {success(bool), stdout, stderr, returncode, timed_out(bool)}
    """
    with tempfile.TemporaryDirectory() as td:
        script_path = os.path.join(td, "script.py")

        # This wrapper captures stdout/stderr and ensures exceptions are caught.
        safe_wrapper = textwrap.dedent("""
            import sys, json, traceback, io, contextlib

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                # Redirect stdout and stderr to capture all output
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
        """) + textwrap.indent(code_str, " " * 12) + textwrap.dedent("""
            except Exception:
                # If any exception occurs during the code execution, capture the traceback
                traceback.print_exc(file=stderr_capture)

            # The final result is printed to the subprocess's stdout as a JSON string
            final_output = {
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue()
            }
            print(json.dumps(final_output))
        """)

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(safe_wrapper)

        # Set environment variables for the subprocess to prevent config errors.
        # MPLCONFIGDIR tells matplotlib where to create its config directory.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "MPLCONFIGDIR": td
        }

        try:
            proc = subprocess.run(
                [sys.executable, "-I", script_path],
                capture_output=True,
                env=env,
                timeout=timeout,
                text=True,
                encoding='utf-8'
            )
            
            try:
                output_json = json.loads(proc.stdout)
                stdout = output_json.get("stdout", "")
                stderr = output_json.get("stderr", "")
            except (json.JSONDecodeError, IndexError):
                stdout = proc.stdout
                stderr = f"JSON PARSING FAILED!\n{proc.stderr}"

            return {
                "success": proc.returncode == 0 and not stderr.strip(),
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
                "returncode": proc.returncode,
                "timed_out": False
            }
        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "stdout": e.stdout or "",
                "stderr": (e.stderr or "") + f"\nExecution timed out after {timeout}s",
                "returncode": None,
                "timed_out": True
            }
        except Exception as exc:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"An unexpected execution error occurred: {repr(exc)}",
                "returncode": None,
                "timed_out": False
            }


# ---------------------------
# 3) Visualization helper
# ---------------------------
def create_visualization(spec: Dict[str, Any], out_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a simple visualization based on `spec`.
    spec example:
      {"chart_type":"bar", "data":{"labels":["a","b"], "values":[1,2]}, "title":"My Chart"}
    Returns: {"success":True, "path":"/tmp/chart.png"} or error dict.
    """
    chart_type = spec.get("chart_type", "bar")
    data = spec.get("data", {})
    title = spec.get("title", "")
    out_path = out_path or os.path.join(tempfile.gettempdir(), f"chart_{int(datetime.now().timestamp())}.png")

    try:
        plt.figure(figsize=(6,4))
        if chart_type == "bar":
            plt.bar(data["labels"], data["values"])
        elif chart_type == "line":
            plt.plot(data["x"], data["y"], marker='o')
        elif chart_type == "scatter":
            plt.scatter(data["x"], data["y"])
        elif chart_type == "pie":
            plt.pie(data["values"], labels=data["labels"], autopct='%1.1f%%')
        else:
            raise ValueError(f"Unsupported chart_type: {chart_type}")

        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return {"success": True, "path": out_path}
    except Exception as exc:
        plt.close()
        return {"success": False, "error": str(exc)}


# ---------------------------
# 4) Executor for plans
# ---------------------------
def execute_plan(plan: List[Dict[str, Any]],
                 code_timeout: int = 15) -> List[Dict[str, Any]]:
    """
    Execute a plan (list of steps) and return outputs per step.
    """
    results = []
    for step in plan:
        sid = step.get("id")
        stype = step.get("type", "note")
        instr = step.get("instruction", "")
        meta = step.get("meta", {})
        result = {"id": sid, "type": stype, "instruction": instr, "meta": meta, "status": "pending", "payload": None}

        try:
            if stype == "code":
                code = meta.get("code", instr)
                exec_result = run_code_sandbox(code, timeout=code_timeout)
                result["status"] = "ok" if exec_result.get("success") else "failed"
                result["payload"] = exec_result
            elif stype == "visualize":
                viz_spec = meta.get("spec", {})
                viz_result = create_visualization(viz_spec)
                result["status"] = "ok" if viz_result.get("success") else "failed"
                result["payload"] = viz_result
            else: # note, api, query or unknown types
                result["status"] = "skipped"
                result["payload"] = {"note": instr, "meta": meta}
        except Exception as exc:
            result["status"] = "failed"
            result["payload"] = {"error": str(exc)}
        
        results.append(result)
    return results

# ---------------------------
# 5) Integration helper: interpret special chat commands
# ---------------------------
def handle_special_commands(prompt: str,
                            state: Dict[str, Any],
                            model_name: str = DEFAULT_MODEL_NAME):
    """
    Processes special CLI commands for the agent:
      - '/plan <task>' -> Decomposes a task into structured steps
      - '/runplan' -> Executes the most recent plan
      - '/exec <python code>' -> Executes Python in a sandbox
    """
    reply = ""
    attachments = []
    handled = False

    if "last_plan" not in state:
        state["last_plan"] = None

    if prompt.startswith("/plan "):
        task = prompt[len("/plan "):].strip()
        plan = decompose_task_via_gemini(task, model_name=model_name)
        state["last_plan"] = plan
        if not plan:
            reply = "⚠️ No plan generated — there may have been an error with the AI model."
        else:
            # We don't need to format the plan here, the UI will do it.
            # The reply will be structured in the app.py file.
            reply = "Plan generated." 
        handled = True

    elif prompt.strip() == "/runplan":
        plan = state.get("last_plan")
        if not plan:
            reply = "⚠️ No plan found. Create one with `/plan <task>` first."
        else:
            reply = "🚀 Executing plan...\n"
            # MERGE ALL CODE into one execution so variables persist
            full_code = "\n\n".join(
                step.get("meta", {}).get("code", "") for step in plan if step.get("type") == "code"
            )
            exec_result = run_code_sandbox(full_code, timeout=30)
            reply += json.dumps(exec_result, indent=2)
        handled = True

    elif prompt.startswith("/exec "):
        code = prompt[len("/exec "):].strip()
        out = run_code_sandbox(code, timeout=10)
        reply = (
            f"🧠 Code executed.\n"
            f"✅ Success: {out.get('success')}\n"
            f"📤 stdout:\n{out.get('stdout')}\n"
            f"⚠️ stderr:\n{out.get('stderr')}"
        )
        handled = True

    return {"handled": handled, "reply": reply, "attachments": attachments, "state": state}

