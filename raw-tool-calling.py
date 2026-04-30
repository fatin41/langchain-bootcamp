import os
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables
load_dotenv()

# 1. Initialize the official Google GenAI Client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "gemini-2.0-flash"

MAX_ITERATIONS = 10

# 2. Define the tools as standard Python functions
# NOTE: Removed @traceable here because it interferes with the SDK's function inspection


def get_product_price(product_name: str) -> str:
    """Returns the price of a product."""
    print(f"> Executing tool: get_product_price for {product_name}")
    prices = {
        "God of War": "INR 499.99",
        "Devil May Cry": "INR 399.99",
        "NFS": "INR 299.99"
    }
    price = prices.get(product_name, "Price not found")
    return f"The price of {product_name} is {price}"


def apply_discount(price: str, discount_tier: str) -> str:
    """Applies a discount to a price string (e.g. 'INR 499.99')."""
    print(
        f"> Executing tool: apply_discount with price={price}, tier={discount_tier}")
    discount_rates = {"silver": 5, "gold": 10, "platinum": 20}
    discount_rate = discount_rates.get(discount_tier.lower(), 0)

    try:
        match = re.search(r"INR\s?([\d.]+)", price)
        if match:
            original_price = float(match.group(1))
        else:
            original_price = float(re.sub(r"[^\d.]", "", price))

        discounted_price = original_price * (100 - discount_rate) / 100
        return f"The discounted price is INR {discounted_price:.2f}"
    except Exception as e:
        return f"Error processing price: {str(e)}"


# Define tools using FunctionDeclaration to be safe and explicit
tools = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_product_price",
                description="Get the price of a product",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "product_name": types.Schema(
                            type="STRING",
                            description="The name of the product to get the price for."
                        )
                    },
                    required=["product_name"]
                )
            ),
            types.FunctionDeclaration(
                name="apply_discount",
                description="Apply a discount to a price",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "price": types.Schema(
                            type="STRING",
                            description="The original price of the product (e.g. 'INR 499.99')"
                        ),
                        "discount_tier": types.Schema(
                            type="STRING",
                            description="The discount tier to apply ('silver', 'gold', 'platinum')"
                        )
                    },
                    required=["price", "discount_tier"]
                )
            )
        ]
    )
]


@traceable(name="Manual Agent Loop")
def run_agent(question: str):
    print(f"Question: {question}")
    print("*" * 60)

    messages = [
        types.Content(role="user", parts=[types.Part(text=question)])
    ]

    config = types.GenerateContentConfig(
        system_instruction="You are a helpful assistant. Use tools when asked about prices or discounts.",
        tools=tools,
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

        # Add model's response to history
        response_content = response.candidates[0].content
        messages.append(response_content)

        # Check for tool calls
        tool_calls = [
            part.function_call for part in response_content.parts if part.function_call]

        if not tool_calls:
            print("No tool calls. Final answer received.")
            return response.text

        # Handle tool calls
        tool_responses = []
        for fc in tool_calls:
            print(f"Model wants to call: {fc.name} with {fc.args}")

            if fc.name == "get_product_price":
                result = get_product_price(**fc.args)
            elif fc.name == "apply_discount":
                result = apply_discount(**fc.args)
            else:
                result = f"Error: Tool {fc.name} not found."

            print(f"Tool Output: {result}")

            tool_responses.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result}
                    )
                )
            )

        # Feed back to model
        messages.append(types.Content(role="user", parts=tool_responses))


def main():
    print("Starting Manual Tool Calling (Raw SDK)...")
    final_answer = run_agent(
        "What is the price of Devil May Cry after a platinum discount?")
    print("\n" + "="*20)
    print("FINAL RESULT:")
    print(final_answer)
    print("="*20)


if __name__ == "__main__":
    main()
