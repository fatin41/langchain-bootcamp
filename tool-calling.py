from pydantic import BaseModel, Field
from typing import List
import re
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

MAX_ITERATIONS = 10


@tool
def get_product_price(product_name: str) -> str:
    """
    A simple tool that returns the price of a product.

    Args:
        product_name: The name of the product to get the price for.

    Returns:
        str: The price of the product.
    """
    print(f"> Executing tool: get_product_price for {product_name}")
    prices = {
        "God of War": "INR 499.99",
        "Devil May Cry": "INR 399.99",
        "NFS": "INR 299.99"
    }
    price = prices.get(product_name, "Price not found")
    return f"The price of {product_name} is {price}"


@tool
def apply_discount(price: str, discount_tier: str) -> str:
    """
    A simple tool that applies a discount to a price.

    Args:
        price: The original price of the product (e.g. 'INR 499.99').
        discount_tier: The discount tier to apply ('silver', 'gold', 'platinum').

    Returns:
        str: The discounted price of the product.
    """
    print(
        f"> Executing tool: apply_discount with price={price}, tier={discount_tier}")
    discount_rates = {"silver": 5, "gold": 10, "platinum": 20}
    discount_rate = discount_rates.get(discount_tier.lower(), 0)

    try:
        # Extract the numeric part of the price

        match = re.search(r"INR\s?([\d.]+)", price)
        if match:
            original_price = float(match.group(1))
        else:
            # Fallback if the model just sends the number
            original_price = float(re.sub(r"[^\d.]", "", price))

        discount_multiplier = (100 - discount_rate) / 100
        discounted_price = original_price * discount_multiplier
        return f"The discounted price is INR {discounted_price:.2f}"
    except Exception as e:
        return f"Error: Could not process price '{price}'. {str(e)}"


@traceable(name="LangSmith Agent Run")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    model_id = "google_genai:gemini-2.5-flash"

    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model_id, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("*" * 60)

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant that answers questions about product prices and applies discounts. "
            "1. ALWAYS use get_product_price to find the price of a product. Never guess. "
            "2. If a discount is requested, use apply_discount ONLY AFTER you have the price. "
            "3. Use the exact price string returned by get_product_price when calling apply_discount."
        )),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # CORRECTED: Use .invoke() instead of calling the object
        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            print("Final Answer generated.")
            return ai_message.content

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"Calling Tool: {tool_name}({tool_args})")

            tool_to_call = tools_dict.get(tool_name)
            if not tool_to_call:
                observation = f"Error: Tool {tool_name} not found."
            else:
                observation = tool_to_call.invoke(tool_args)

            print(f"Observation: {observation}")

            # CRITICAL: Feed the tool result back to the LLM
            messages.append(ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            ))


def main():
    print("Starting Tool Calling Lesson...")
    result = run_agent(
        "What is the price of Devil May Cry after platinum discount?")
    print("\n" + "="*20)
    print("FINAL RESULT:")
    print(result)
    print("="*20)


if __name__ == "__main__":
    main()
