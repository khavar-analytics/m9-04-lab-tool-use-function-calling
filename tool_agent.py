import json
import re
import ollama

# ── Load orders ──────────────────────────────────────────────
with open("orders.json") as f:
    ORDERS = json.load(f)

# ── Tool functions ────────────────────────────────────────────
def lookup_order(order_id: str) -> dict:
    order = ORDERS.get(order_id)
    if not order:
        return {"error": f"Order '{order_id}' not found."}
    return order

def calculate(expression: str) -> dict:
    try:
        allowed = re.fullmatch(r"[\d\s\+\-\*\/\.\(\)]+", expression)
        if not allowed:
            return {"error": "Invalid expression. Use only numbers and operators: + - * /"}
        result = eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# ── Tool schemas ──────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up an order by its ID. "
                "Returns item name, price, purchase date, and warranty length in months. "
                "Always call this FIRST before doing any math."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. A1001"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a simple arithmetic expression and return the numeric result. "
                "ONLY pass plain numbers and operators like '1200 * 3'. "
                "Never pass function calls or variable names — only raw numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Plain math only, e.g. '1200 * 3' or '250 + 80'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ── System prompt ─────────────────────────────────────────────
SYSTEM = """You are an order assistant. You have two tools:
1. lookup_order(order_id) — call this FIRST to get the price of an order.
2. calculate(expression) — call this SECOND with plain numbers only, e.g. '1200 * 3'.

IMPORTANT rules:
- Never put function calls inside calculate(). Only use raw numbers.
- Always lookup the order first, extract the price number, then pass that number to calculate.
- Do not guess prices. Always use the tool result.
"""

# ── Tool dispatcher ───────────────────────────────────────────
def dispatch_tool(name: str, args: dict) -> str:
    print(f"Tool called : {name}")
    print(f"Arguments   : {args}")

    if name == "lookup_order":
        result = lookup_order(**args)
    elif name == "calculate":
        result = calculate(**args)
    else:
        result = {"error": f"Unknown tool: {name}"}

    print(f"Tool result : {result}")
    return json.dumps(result)

# ── Agentic loop ──────────────────────────────────────────────
def run_agent(question: str, model: str = "qwen2.5:7b"):
    print(f"\n{'='*60}")
    print(f"{question}")
    print('='*60)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": question}
    ]

    while True:
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOLS
        )

        msg = response["message"]

        if not msg.get("tool_calls"):
            print(f"\nFinal answer:\n{msg['content']}\n")
            break

        messages.append(msg)

        for tc in msg["tool_calls"]:
            tool_name = tc["function"]["name"]
            tool_args = tc["function"]["arguments"]

            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)

            tool_result = dispatch_tool(tool_name, tool_args)

            messages.append({
                "role": "tool",
                "content": tool_result
            })

# ── Scenarios ─────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Both tools in sequence
    run_agent("For order A1001, what would the total cost be if I bought three of them?")

    # 2. No tool needed
    run_agent("What can you help me with?")

    # 3. Stretch: non-existent order
    run_agent("Can you look up order A9999 for me?")