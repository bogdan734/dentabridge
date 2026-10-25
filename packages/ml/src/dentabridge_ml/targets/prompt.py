"""The one instruction used for training, zero-shot baselines and the shipped model.

Training and inference must share this text, or a baseline is not comparable to the tuned model.
"""

from dentabridge_core import FindingCode, Modality

VIEW_NAME: dict[Modality, str] = {
    "intraoral_photo": "photo of a person's teeth and gums",
    "panoramic_xray": "panoramic dental X-ray",
}

SCHEMA = (
    '{"image_quality": "ok" | "unusable", '
    '"quality_reason": null | "blurry" | "dark" | "no_teeth" | "not_dental", '
    '"findings": [{"code": string, "status": "possible" | "not_seen", '
    '"regions": [{"x0": number, "y0": number, "x1": number, "y1": number}]}]}'
)


def build_prompt(assess: frozenset[FindingCode], modality: Modality) -> str:
    """Ask about exactly the codes in `assess`, listed in catalogue order."""
    if not assess:
        raise ValueError("a prompt needs at least one finding to assess")
    codes = ", ".join(code.value for code in FindingCode if code in assess)
    return (
        f"This is a {VIEW_NAME[modality]}. Reply with JSON only, matching this schema: {SCHEMA}. "
        f"Judge only these findings: {codes}. "
        '"possible" means a region that may show the finding, given as boxes in image fractions '
        '(0 to 1, origin top left); "not_seen" means none was noticed. '
        'If the image cannot be judged, set image_quality to "unusable", give the reason and '
        "return no findings."
    )
