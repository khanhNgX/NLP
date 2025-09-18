from typing import List

def load_raw_text_data(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines
