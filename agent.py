import json
import os
import subprocess
import time
import logging
from openai import AzureOpenAI
from azure_llm import build_llm_config

logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client with configuration from Key Vault
llm_config = build_llm_config(temperature=0)
client = AzureOpenAI(
    api_key=llm_config['api_key'],
    api_version=llm_config['api_version'],
    base_url=f"{llm_config['endpoint']}/openai/deployments/{llm_config['model']}"
)

# Define the paths to our sandbox files
CODE_FILE = "calc.py"
TEST_FILE = "test_calc.py"

# ==========================================
# 1. DEFINE THE TOOLS (THE AGENT'S HANDS)
# ==========================================

def run_tests() -> str:
    """Executes pytest on the test file and returns stdout and stderr combined."""
    result = subprocess.run(["pytest", TEST_FILE], capture_output=True, text=True)
    # Combine output so the LLM sees the exact traceback
    return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

def write_code(content: str) -> str:
    """Overwrites the target application source file with new content."""
    with open(CODE_FILE, "w") as f:
        f.write(content)
    return f"Successfully updated {CODE_FILE}"

# Tool metadata schemas mapping for the OpenAI API
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the pytest suite against the application to check for errors."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_code",
            "description": "Overwrite the calc.py file with updated Python source code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The entire raw string content of the python file."}
                },
                "required": ["content"]
            }
        }
    }
]

# ==========================================
# 2. THE CORE ORCHESTRATION LOOP
# ==========================================

def run_self_healing_loop(max_iterations: int = 6):
    print("🚀 Initializing Self-Healing Agent...")
    
    # Maintain conversation memory state
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous debugging agent. Your goal is to make the tests pass. "
                f"You are modifying '{CODE_FILE}'. Run the tests first to discover errors, "
                "analyze the traceback, rewrite the code to fix the bug, and repeat until "
                "the suite runs cleanly. Do not stop until tests yield a 100% success rate."
            )
        },
        {
            "role": "user",
            "content": f"Please fix the bug in '{CODE_FILE}' so that '{TEST_FILE}' passes successfully."
        }
    ]

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- 🔄 ITERATION {iteration} ---")
        
        # 1. Ask the model what to do next with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=llm_config['model'],
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    timeout=llm_config.get('timeout', 180)
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 0.5
                    print(f"⚠️ Azure LLM error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed to get response from Azure LLM after {max_retries} attempts")
                    raise
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        # Print the thought process if the model provided text
        if response_message.content:
            print(f"🤖 Agent Thought: {response_message.content}")

        # 2. Check if the model decided to call a tool
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                print(f"🛠️ Executing Tool: {tool_name}")
                
                # Route to the appropriate local Python function
                if tool_name == "run_tests":
                    tool_output = run_tests()
                    # Simple heuristic: look for success signals in pytest output
                    if "passed" in tool_output.lower() and "failed" not in tool_output.lower():
                        print("✅ Success! All tests passed cleanly.")
                        print(tool_output)
                        return True
                    else:
                        print("❌ Tests Failed. Feeding traceback details back to the agent.")
                        
                elif tool_name == "write_code":
                    tool_output = write_code(tool_args["content"])
                    print(f"📝 Modified {CODE_FILE}")
                
                # 3. Append the execution result back to the conversation thread
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_output
                })
        else:
            print("🤖 Agent stopped making tool calls.")
            break
            
    print("\n⚠️ Agent reached the iteration threshold without resolving the failure.")
    return False

if __name__ == "__main__":
    run_self_healing_loop()