from transformers import pipeline

# Load zero-shot classification model locally (no API needed)
# facebook/bart-large-mnli is the most accurate zero-shot classifier
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

CANDIDATE_LABELS = [
    "http status code response",
    "security alert or intrusion",
    "critical system error",
    "general error",
    "resource usage or capacity warning",
    "user action or activity",
    "system notification or maintenance",
    "workflow error",
    "deprecation warning",
    "network issue",
]
HYPOTHESIS_TEMPLATE = "This log entry is about {}."
LABEL_MAPPING = {
    "http status code response": "HTTP Status",
    "security alert or intrusion": "Security Alert",
    "critical system error": "Critical Error",
    "general error": "Error",
    "resource usage or capacity warning": "Resource Usage",
    "user action or activity": "User Action",
    "system notification or maintenance": "System Notification",
    "workflow error": "Workflow Error",
    "deprecation warning": "Deprecation Warning",
    "network issue": "Network Issue",
}


def classify_with_llm_with_confidence(log_msg):
    """
    Classify log messages using a local zero-shot NLI model and return both
    the label and the confidence score.
    """
    try:
        result = classifier(
            log_msg,
            CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            multi_label=True
        )
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])

        # Use 0.5 threshold to filter out unrelated logs (Unclassified)
        if top_score < 0.5:
            return "Unclassified", top_score

        return LABEL_MAPPING[top_label], top_score
    except Exception as e:
        print(f"Error during local NLI classification: {e}")
        return "Unclassified", 0.0


def classify_with_llm(log_msg):
    """
    Classify log messages using a local zero-shot NLI model.
    No API key or internet connection needed at inference time.

    Categories: HTTP Status, Security Alert, Critical Error, Error,
    Resource Usage, User Action, System Notification, Workflow Error,
    Deprecation Warning, Network Issue, or Unclassified.
    """
    label, _ = classify_with_llm_with_confidence(log_msg)
    return label


if __name__ == "__main__":
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