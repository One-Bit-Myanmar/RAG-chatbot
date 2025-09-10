import re
import spacy

nlp = spacy.load("en_core_web_sm")

email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
password_pattern = re.compile(r"(?i)\bpassword\s*(?:=|:|is)?\s*\S+")
def redact_pii(text):
    doc = nlp(text)
    redacted_text = ""
    last_end = 0

    for ent in doc.ents:
        redacted_text += text[last_end:ent.start_char]
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            redacted_text += "[REDACTED]"
        else:
            redacted_text += ent.text
        last_end = ent.end_char

    redacted_text += text[last_end:]
    redacted_text = email_pattern.sub("[REDACTED_EMAIL]", redacted_text)
    redacted_text = password_pattern.sub("[REDACTED_PASSWORD]", redacted_text)

    return redacted_text


text = """
My account got hack. My email is Linemaygakt@gmail.com and password is testtest123!! . how to get back my account.
"""

print(redact_pii(text))
