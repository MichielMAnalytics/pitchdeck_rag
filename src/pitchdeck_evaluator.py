# src/pitchdeck_evaluator.py
import openai
import json
import re
import os

# You'll need to pass the OpenAI API key to this function or ensure it's set
# as an environment variable in the context where this function is called.
# For Streamlit app, we'll pass it from app.py.

def evaluate_startup(all_descriptions: str, uploaded_file_name: str, openai_api_key: str):
    """
    Evaluates the startup based on combined slide descriptions against predefined conditions.
    This function is designed to be called from the main app.
    """
    # Ensure the OpenAI API key is set for this context
    openai.api_key = openai_api_key

    startup_name_eval = os.path.splitext(uploaded_file_name)[0].replace("_", " ").replace("-", " ").title()

    # Updated conditions and response format based on user's request
    conditions_prompt_text = f"""
Based on the description of the startup from its pitch deck slides, please answer **each** of the following criteria:
1.  **Funding Round**: What funding round is the startup seeking or currently in (e.g., Seed, Series A, Series B, etc.)? If not explicitly stated, infer based on the stage (e.g., early traction suggests Seed/Pre-Seed, significant revenue suggests Series A/B).
2.  **Region**: What is the primary geographical region or target market of the startup (e.g., San Francisco, Europe, Global, specific countries)?
3.  **Category**: What is the primary industry or category of the startup (e.g., SaaS, FinTech, AI, Healthcare, E-commerce, Deep Tech)?
4.  **Excluded Fields**: List whether the startup is active in any of the following fields: crypto development, cryptocurrencies, or drug development. If none are mentioned, state "None explicitly mentioned."
"""
    format_response_prompt_text = f"""
Provide your response in **valid and complete JSON format** with the following structure.
Ensure the JSON is parseable:
{{
    "startup_name": "{startup_name_eval}",
    "funding_round": "string (e.g., Seed, Series A, Series B, Not specified, Unknown)",
    "region": "string (e.g., San Francisco, Europe, Global, Germany, None explicitly mentioned)",
    "category": "string (e.g., SaaS, FinTech, AI, Healthcare, E-commerce, Deep Tech, Not specified, Unknown)",
    "excluded_fields": "string (e.g., crypto development, cryptocurrencies, drug development, None explicitly mentioned. If multiple, list all comma-separated)"
}}
"""
    eval_prompt_messages = [
        {"role": "system", "content": "You are a venture capital investment manager evaluating startups based on the provided text from their pitch deck slides. Strictly follow the JSON output format requested. Ensure the entire response is valid JSON."},
        {"role": "user", "content": f"Here is the combined content from the startup's pitch deck slides:\n\n---------------------\n{all_descriptions}\n---------------------\n\n{conditions_prompt_text}\n\n{format_response_prompt_text}"}
    ]
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o", # Use the same model as app.py or make configurable
            messages=eval_prompt_messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
        raw_llm_response_eval = completion.choices[0].message.content
        try:
            conditions_dict = json.loads(raw_llm_response_eval)
            return conditions_dict
        except json.JSONDecodeError as e_json:
            print(f"Warning: Direct JSON parsing failed ({e_json}), attempting regex extraction.")
            match = re.search(r"```json\s*(.*?)\s*```", raw_llm_response_eval, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_content = match.group(1)
                try:
                    conditions_dict = json.loads(extracted_content)
                    return conditions_dict
                except json.JSONDecodeError as e_re:
                    print(f"Error: Failed to parse JSON from evaluation response even after regex: {e_re}")
                    return {"error": "Failed to parse JSON after regex", "raw_llm_response_eval": raw_llm_response_eval}
            else:
                print("Error: Failed to parse JSON and no JSON code block found in response.")
                return {"error": "No JSON code block found", "raw_llm_response_eval": raw_llm_response_eval}
    except Exception as e_eval:
        print(f"Error during startup evaluation API call: {e_eval}")
        return {"error": str(e_eval)}

# Example usage (for testing this module independently)
if __name__ == "__main__":
    # This block will not be executed by the Streamlit app, but is useful for testing
    print("Running pitchdeck_evaluator.py as main for testing.")
    test_api_key = os.environ.get("OPENAI_API_KEY")
    if not test_api_key:
        print("OPENAI_API_KEY environment variable not set. Cannot run standalone test.")
    else:
        dummy_descriptions = """
        Content from Slide 1: Our startup, Crypto Innovations, is building a new decentralized finance (DeFi) platform. We are seeking a Seed round to expand our team in Berlin.
        Content from Slide 2: Our target market is primarily Europe, with a focus on Germany and France. We are an innovative FinTech company leveraging blockchain technology.
        Content from Slide 3: We aim to disrupt traditional banking with secure and transparent cryptocurrency transactions. We avoid drug development.
        """
        dummy_file_name = "crypto_innovations_pitchdeck.pdf"

        print("\n--- Running Evaluation Test ---")
        results = evaluate_startup(dummy_descriptions, dummy_file_name, test_api_key)

        if isinstance(results, dict) and "error" in results:
            print(f"Evaluation failed: {results.get('error')}")
            if "raw_llm_response_eval" in results:
                print(f"Raw LLM Response:\n{results['raw_llm_response_eval']}")
        else:
            print("Evaluation Results:")
            print(json.dumps(results, indent=4))

        print("\n--- Running Another Evaluation Test (No Excluded Fields) ---")
        dummy_descriptions_2 = """
        Content from Slide 1: Our company, SaaS Pro, is developing an Enterprise SaaS solution for supply chain optimization. We are looking for Series A funding.
        Content from Slide 2: Our primary market is North America, specifically the US and Canada. We have strong traction and significant revenue.
        Content from Slide 3: We are revolutionizing logistics with AI-powered analytics. We are not involved in crypto or drug development.
        """
        dummy_file_name_2 = "saas_pro_pitchdeck.pdf"
        results_2 = evaluate_startup(dummy_descriptions_2, dummy_file_name_2, test_api_key)
        if isinstance(results_2, dict) and "error" in results_2:
            print(f"Evaluation failed: {results_2.get('error')}")
            if "raw_llm_response_eval" in results_2:
                print(f"Raw LLM Response:\n{results_2['raw_llm_response_eval']}")
        else:
            print("Evaluation Results:")
            print(json.dumps(results_2, indent=4))