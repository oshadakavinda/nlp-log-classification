import os
from dotenv import load_dotenv
from groq import Groq
import json
import re


load_dotenv()

_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        return None
        
    try:
        _groq_client = Groq(api_key=api_key)
        return _groq_client
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return None


def classify_with_llm(log_msg):
    """
    Generate a variant of the input sentence. For example,
    If input sentence is "User session timed out unexpectedly, user ID: 9250.",
    variant would be "Session timed out for user 9251"
    """
    groq = get_groq_client()
    if not groq:
        return "Unclassified"

    prompt = f'''Classify the log message into one of these categories: 
    (1) Workflow Error, (2) Deprecation Warning.
    If you can't figure out a category, use "Unclassified".
    Put the category inside <category> </category> tags. 
    Log message: {log_msg}'''

    try:
        chat_completion = groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            # model="llama-3.3-70b-versatile",
            model="deepseek-r1-distill-llama-70b",
            temperature=0.5
        )

        content = chat_completion.choices[0].message.content
        match = re.search(r'<category>(.*)<\/category>', content, flags=re.DOTALL)
        category = "Unclassified"
        if match:
            category = match.group(1).strip()

        return category
    except Exception as e:
        print(f"Error during LLM classification: {e}")
        return "Unclassified"


if __name__ == "__main__":
    print(classify_with_llm(
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    print(classify_with_llm(
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
    print(classify_with_llm("System reboot initiated by user 12345."))