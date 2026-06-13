import re
import time
from sqlmodel import Session, select
from app.database import engine
from app.models.regex_rule import RegexRule

_regex_cache = None
_regex_last_fetched = None

# Default hardcoded rules to use as fallback if database is empty
DEFAULT_RULES = [
    (r"User User\d+ logged (in|out).", "User Action"),
    (r"Backup (started|ended) at .*", "System Notification"),
    (r"Backup completed successfully.", "System Notification"),
    (r"System updated to version .*", "System Notification"),
    (r"File .* uploaded successfully by user .*", "System Notification"),
    (r"Disk cleanup completed successfully.", "System Notification"),
    (r"System reboot initiated by user .*", "System Notification"),
    (r"Account with ID .* created by .*", "User Action")
]


def get_regex_rules():
    global _regex_cache, _regex_last_fetched
    now = time.time()
    
    # Refresh cache every 5 seconds
    if _regex_cache is None or _regex_last_fetched is None or now - _regex_last_fetched > 5:
        try:
            with Session(engine) as session:
                rules = session.exec(select(RegexRule)).all()
                if rules:
                    compiled = []
                    for r in rules:
                        try:
                            compiled.append((re.compile(r.pattern), r.target_label))
                        except Exception as parse_error:
                            print(f"[Regex] Skip invalid pattern '{r.pattern}': {parse_error}")
                    _regex_cache = compiled
                else:
                    # Database is empty, use defaults
                    _regex_cache = [(re.compile(pat), label) for pat, label in DEFAULT_RULES]
                _regex_last_fetched = now
        except Exception as e:
            print(f"[Regex] Error loading rules from database: {e}")
            # Fall back to default rules if database fails
            if _regex_cache is None:
                _regex_cache = [(re.compile(pat), label) for pat, label in DEFAULT_RULES]
                
    return _regex_cache


def classify_with_regex(log_message):
    rules = get_regex_rules()
    for compiled_re, label in rules:
        if compiled_re.search(log_message):
            return label
    return None


if __name__ == "__main__":
    print(classify_with_regex("Backup completed successfully."))
    print(classify_with_regex("Account with ID 1234 created by User1."))
