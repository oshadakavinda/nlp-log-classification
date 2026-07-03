import httpx
import json
from app.core.config import settings

# Candidate labels for log classification
CANDIDATE_LABELS = [
    "HTTP Status",
    "Security Alert",
    "Critical Error",
    "Error",
    "Resource Usage",
    "User Action",
    "System Notification",
    "Workflow Error",
    "Deprecation Warning",
    "Network Issue",
    "Unclassified"
]

def classify_with_llm_with_confidence(log_msg):
    """
    Classify log messages using Google Gemini API zero-shot classification and return
    the predicted label and confidence score.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        print("[Gemini] WARNING: GEMINI_API_KEY is not configured in your .env file. LLM fallback skipped.")
        return "Unclassified", 0.0

    model = settings.GEMINI_MODEL or "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    prompt = (
        f"You are a log classifier. Analyze the following log message and select the most appropriate category "
        f"from the allowed list: {CANDIDATE_LABELS}.\n\n"
        f"Log Message: \"{log_msg}\"\n\n"
        f"Return the classification as a JSON object matching the requested schema."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "label": {
                        "type": "STRING",
                        "enum": CANDIDATE_LABELS,
                        "description": "The category of the log message"
                    },
                    "confidence": {
                        "type": "NUMBER",
                        "description": "Confidence score between 0.0 and 1.0"
                    }
                },
                "required": ["label", "confidence"]
            }
        }
    }

    try:
        response = httpx.post(url, json=payload, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        response_text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(response_text)
        
        label = result.get("label", "Unclassified")
        confidence = float(result.get("confidence", 0.0))
        
        return label, confidence
    except Exception as e:
        print(f"[Gemini] Error during API classification: {e}")
        return "Unclassified", 0.0


def classify_with_llm(log_msg):
    """
    Classify log messages using the Gemini API.
    """
    label, _ = classify_with_llm_with_confidence(log_msg)
    return label


if __name__ == "__main__":
    # Test block
    test_logs = [
        "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active.",
        "The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025",
        "System reboot initiated by user 12345.",
        "Invoice generation process aborted for order ID 8910 due to invalid tax calculation module.",
        "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' for improved functionality.",
        "Lead conversion failed for prospect ID 7842 due to missing contact information.",
        "API endpoint 'getCustomerDetails' is deprecated and will be removed in version 3.2. Use 'fetchCustomerInfo' instead.",
    ]
    for log in test_logs:
        label = classify_with_llm(log)
        print(f"{label:25s} <- {log[:80]}")