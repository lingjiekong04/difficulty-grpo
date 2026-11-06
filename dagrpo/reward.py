"""One strict numeric scoring rule shared by training and evaluation."""
import re
from decimal import Decimal, InvalidOperation

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
FINAL = re.compile(r"####\s*(" + NUMBER + r")\s*\Z")

def number(value):
    text = str(value).strip()
    if len(text) > 128 or not re.fullmatch(NUMBER, text):
        raise ValueError("Invalid numeric answer")
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation as exc:
        raise ValueError("Invalid numeric answer") from exc

def score(text, truth, truncated=False):
    expected = number(truth)  # corrupt ground truth must fail loudly
    match = FINAL.search(text.strip())
    valid = match is not None and text.count("####") == 1
    correct = valid and number(match.group(1)) == expected
    return {"score": float(correct and not truncated),
            "acc": float(correct and not truncated),
            "format_error": float(not valid), "truncated": float(truncated)}

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    info = extra_info or {}
    if "dagrpo_truncated" not in info:
        raise ValueError("Use StrictNumericRewardManager: missing termination metadata")
    return score(solution_str, ground_truth, info["dagrpo_truncated"])
