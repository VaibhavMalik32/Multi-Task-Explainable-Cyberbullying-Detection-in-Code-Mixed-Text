import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def extract_rationale(text):
    words = re.findall(r"\{(.*?)\}", str(text))
    return words