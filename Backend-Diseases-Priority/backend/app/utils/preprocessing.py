import re

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.replace(";", ",") 
    text = re.sub(r"[^a-zA-Z0-9, ]", "", text)
    return text.strip()
