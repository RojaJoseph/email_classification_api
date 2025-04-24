import re

def mask_pii(text: str) -> str:
    if not isinstance(text, str):
        return text

    # Mask email addresses
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}', '[EMAIL]', text)

    # Mask phone numbers (general pattern)
    text = re.sub(r'\b(\+?\d{1,3}[\s.-]?)?\(?\d{3,5}\)?[\s.-]?\d{3,5}[\s.-]?\d{3,5}\b', '[PHONE]', text)

    # Mask simple names (assume capitalized first/last names — very basic)
    text = re.sub(r'\b([A-Z][a-z]{2,})\s([A-Z][a-z]{2,})\b', '[NAME]', text)

    return text
