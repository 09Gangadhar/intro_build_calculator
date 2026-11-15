import math
import os
from getpass import getpass

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = getpass("Groq API key: ").strip()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


# 1. Define the calculator tools
@tool
def add_apples(a: float, b: float) -> float:
  """Add two numbers of apples together."""
  return a + b


@tool
def tree_distance(height_above_eye_m: float, angle_degrees: float) -> float:
  """Calculate horizontal distance to a tree given height above eye level (in metres) and angle of elevation (in degrees)."""
  return round(height_above_eye_m / math.tan(math.radians(angle_degrees)), 2)


# 2. Initialize model and agent
model = ChatGroq(
    model="openai/gpt-oss-20b", temperature=0, timeout=30, max_retries=1
)

agent = create_agent(
    model=model,
    tools=[add_apples, tree_distance],
    system_prompt=(
        "You are an intelligent calculator. Use add_apples for apple addition"
        " questions and tree_distance for tree trigonometry questions. If a"
        " tree-distance question is missing the angle, ask the user for it."
    ),
)

# 3. Expose the agent over MCP
mcp = FastMCP(
    "Intelligent Calculator",
    host=os.environ.get("INTRO_MCP_HOST", "127.0.0.1"),
    stateless_http=True,
    json_response=True,
)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def solve_math_problem(problem: str) -> str:
  """Solve an apples addition or tree-distance word problem. Pass the original wording."""
  problem = problem.strip()
  if not 1 <= len(problem) <= 1500:
    return "Enter an English word problem of 1 to 1500 characters."
  result = agent.invoke(
      {"messages": [("human", problem)]},
      config={"recursion_limit": 8, "run_name": "intro_calculator_mcp"},
  )
  return result["messages"][-1].text


if __name__ == "__main__":
  mcp.run(transport="streamable-http")
