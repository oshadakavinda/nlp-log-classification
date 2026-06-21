import re
from app.services.processor_regex import classify_with_regex
from app.services.processor_bert import classify_with_bert, classify_with_bert_with_confidence
from app.services.processor_llm import classify_with_llm, classify_with_llm_with_confidence

def infer_source(log_msg: str) -> str:
    """
    Dynamically infer a plausible source name from the log message content
    if no source was provided or if the source is 'Unknown'.
    """
    log_msg_lower = log_msg.lower()
    
    # Security must be checked BEFORE database, since 'sql injection' contains 'sql'
    if "security" in log_msg_lower or "injection" in log_msg_lower or "rce" in log_msg_lower or "attack" in log_msg_lower or "unauthorized" in log_msg_lower or "privilege escalation" in log_msg_lower or "cors" in log_msg_lower or "firewall" in log_msg_lower or "blocked" in log_msg_lower or "suspicious" in log_msg_lower:
        return "SecurityMonitor"
    if "db" in log_msg_lower or "database" in log_msg_lower or "postgres" in log_msg_lower or "sql" in log_msg_lower or "connection pool" in log_msg_lower:
        return "DatabasePool"
    if "auth" in log_msg_lower or "login" in log_msg_lower or "token" in log_msg_lower or "credential" in log_msg_lower or "session" in log_msg_lower:
        return "AuthService"
    if "payment" in log_msg_lower or "checkout" in log_msg_lower or "cart" in log_msg_lower or "transaction" in log_msg_lower:
        return "PaymentGateway"
    if "api" in log_msg_lower or "get /" in log_msg_lower or "post /" in log_msg_lower or "put /" in log_msg_lower or "delete /" in log_msg_lower:
        return "APIGateway"
    if "memory" in log_msg_lower or "heap" in log_msg_lower or "exhausted" in log_msg_lower or "disk" in log_msg_lower or "io error" in log_msg_lower:
        return "SystemAgent"
    if "download" in log_msg_lower or "upload" in log_msg_lower or "pdf" in log_msg_lower or "file" in log_msg_lower:
        return "FileService"
    if "deploy" in log_msg_lower or "container" in log_msg_lower or "replicas" in log_msg_lower or "scaled" in log_msg_lower or "microservice" in log_msg_lower or "communication failure" in log_msg_lower or "srv-" in log_msg_lower:
        return "Orchestrator"
    if "live" in log_msg_lower or "health check" in log_msg_lower or "load-balancer" in log_msg_lower or "load balancer" in log_msg_lower:
        return "LoadBalancer"
    
    class_match = re.search(r'([a-zA-Z0-9_]+\.)+([a-zA-Z0-9_]+Service)', log_msg)
    if class_match:
        return class_match.group(2)
        
    return "System"


def classify(logs):
    labels = []
    for source, log_msg in logs:
        label, _, _ = classify_log(source, log_msg)
        labels.append(label)
    return labels


def classify_log(source, log_msg):
    label = classify_with_regex(log_msg)
    if label:
        return label, "Regex", 1.0
    label, confidence = classify_with_bert_with_confidence(log_msg)
    if label == "Unclassified" or confidence < 0.5:
        # Fallback to local zero-shot NLI (LLM)
        fallback_label, fallback_conf = classify_with_llm_with_confidence(log_msg)
        if fallback_label != "Unclassified":
            return fallback_label, "LLM", fallback_conf
    return label, "ML", confidence

def classify_csv(input_file):
    import pandas as pd
    df = pd.read_csv(input_file)

    # Perform classification
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

    # Save the modified file
    output_file = "output.csv"
    df.to_csv(output_file, index=False)

    return output_file

if __name__ == '__main__':
    classify_csv("test.csv")
    # logs = [
    #     ("ModernCRM", "IP 192.168.133.114 blocked due to potential attack"),
    #     ("BillingSystem", "User 12345 logged in."),
    #     ("AnalyticsEngine", "File data_6957.csv uploaded successfully by user User265."),
    #     ("AnalyticsEngine", "Backup completed successfully."),
    #     ("ModernHR", "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1 RCODE  200 len: 1583 time: 0.1878400"),
    #     ("ModernHR", "Admin access escalation detected for user 9429"),
    #     ("LegacyCRM", "Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."),
    #     ("LegacyCRM", "Invoice generation process aborted for order ID 8910 due to invalid tax calculation module."),
    #     ("LegacyCRM", "The 'BulkEmailSender' feature is no longer supported. Use 'EmailCampaignManager' for improved functionality."),
    #     ("LegacyCRM", " The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025")
    # ]
    # labels = classify(logs)
    #
    # for log, label in zip(logs, labels):
    #     print(log[0], "->", label)


