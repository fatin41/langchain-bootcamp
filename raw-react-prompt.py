import os
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Initialize the Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "gemini-2.0-flash"

MAX_ITERATIONS = 10

# 2. Define the tools


def get_product_price(product_name: str) -> str:
    """Returns the price of a product from the database."""
    print(f"> Executing tool: get_product_price for {product_name}")
    prices = {
        "God of War": "INR 499.99",
        "Devil May Cry": "INR 399.99",
        "NFS": "INR 299.99"
    }
    price = prices.get(product_name, "Price not found")
    return f"The price of {product_name} is {price}"


def apply_discount(price: str, discount_tier: str) -> str:
    """Applies a percentage discount (silver: 5%, gold: 10%, platinum: 20%) to a price string."""
    print(
        f"> Executing tool: apply_discount with price={price}, tier={discount_tier}")
    discount_rates = {"silver": 5, "gold": 10, "platinum": 20}
    discount_rate = discount_rates.get(discount_tier.lower(), 0)

    try:
        # Extract numeric value using regex
        match = re.search(r"INR\s?([\d.]+)", price)
        original_price = float(match.group(1)) if match else float(
            re.sub(r"[^\d.]", "", price))

        discounted_price = original_price * (100 - discount_rate) / 100
        return f"The discounted price is INR {discounted_price:.2f}"
    except Exception as e:
        return f"Error processing price: {str(e)}"


# Mapping for the manual execution loop
tools_map = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

# 3. The ReAct System Prompt
# This is the "brain surgery" that turns an LLM into an Agent.
SYSTEM_INSTRUCTIONS = """
You are a ReAct Agent. You solve problems by alternating between steps: Thought, Action, and Observation.

1. **Thought**: Explain what you know and what you need to do next.
2. **Action**: Call a tool to get new information.
3. **Observation**: Review the data returned by the tool.

RULES:
- Always find the price of a product using 'get_product_price' BEFORE applying a discount.
- Do not make up prices. If a tool says 'Price not found', tell the user.
- Once you have the final answer, provide it clearly to the user.
"""


def run_agent(question: str):
    print(f"Question: {question}")
    print("*" * 60)

    # Initialize conversation history with the system instruction
    messages = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTIONS,
        tools=list(tools_map.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True)
    )

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=messages,
            config=config
        )

        response_content = response.candidates[0].content

        # Log the model's "Thought" (text response)
        if response.text:
            print(f"AI Thought: {response.text.strip()}")

        messages.append(response_content)

        # Check if the model wants to call a tool (The "Action")
        tool_calls = [
            part.function_call for part in response_content.parts if part.function_call]

        if not tool_calls:
            print("No more tool calls needed. Goal reached.")
            return response.text

        tool_responses = []
        for fc in tool_calls:
            print(f"Action: Calling {fc.name} with {fc.args}")

            # Execute the actual Python function
            tool_func = tools_map.get(fc.name)
            if tool_func:
                result = tool_func(**fc.args)
            else:
                result = f"Error: Tool {fc.name} not found."

            print(f"Observation: {result}")

            # Wrap the result as a FunctionResponse for the LLM
            tool_responses.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result}
                    )
                )
            )

        # Add the "Observation" back to the message history
        messages.append(types.Content(role="user", parts=tool_responses))


def main():
    print("Starting Manual ReAct Agent...")
    final_answer = run_agent(
        "What is the price of Devil May Cry after a platinum discount?")
    print("\n" + "="*20)
    print("FINAL RESULT:")
    print(final_answer)
    print("="*20)


if __name__ == "__main__":
    main()
