"""
Entity Resolution Preprocessing Module
Cleans business names and addresses, normalizes text, and extracts features.
"""

import re

# Common business suffixes to normalize or strip during token comparison
LEGAL_SUFFIXES = {
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'gmbh', 'sa', 'sas', 'srl', 'plc', 'bv', 'nv', 'enterprises',
    'group', 'holdings', 'services', 'solutions', 'technologies', 'intl', 'international'
}

# Common address abbreviations
ADDRESS_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'ste': 'suite',
    'apt': 'apartment', 'bldg': 'building', 'fl': 'floor', 'pkwy': 'parkway',
    'hwy': 'highway', 'sq': 'square', 'pl': 'place', 'ctr': 'center'
}


def clean_text(text: str) -> str:
    """Basic text normalization: lowercase, strip special chars, normalize whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    # Replace non-alphanumeric (except spaces) with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_name(name: str) -> str:
    """Normalize business name by cleaning and standardizing legal terms."""
    text = clean_text(name)
    tokens = text.split()
    # Filter out sole legal suffixes for cleaner matching representation
    filtered = [t for t in tokens if t not in LEGAL_SUFFIXES]
    return " ".join(filtered) if filtered else text


def clean_address(addr: str) -> str:
    """Normalize business address by cleaning and standardizing common street terms."""
    text = clean_text(addr)
    tokens = text.split()
    normalized = [ADDRESS_ABBR.get(t, t) for t in tokens]
    return " ".join(normalized)


def extract_digits(text: str) -> list:
    """Extract numeric sequences (PIN codes, street numbers, door numbers)."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\d+\b', text)
