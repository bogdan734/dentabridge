"""What the model is shown: one instruction and one picture size, for training, zero-shot
baselines and the shipped model.

Training and inference must share both, or a baseline is not comparable to the tuned model.
"""

from PIL import Image

from dentabridge_core import FindingCode, Modality

VIEW_NAME: dict[Modality, str] = {
    "intraoral_photo": "photo of a person's teeth and gums",
    "panoramic_xray": "panoramic dental X-ray",
}

SCHEMA = (
    '{"image_quality": "ok" | "unusable", '
    '"quality_reason": null | "blurry" | "dark" | "no_teeth" | "not_dental", '
    '"findings": [{"code": string, "status": "possible" | "not_seen", '
    '"regions": [[x0, y0, x1, y1]]}]}'
)


def build_prompt(assess: frozenset[FindingCode], modality: Modality) -> str:
    """Ask about exactly the codes in `assess`, listed in catalogue order."""
    if not assess:
        raise ValueError("a prompt needs at least one finding to assess")
    codes = ", ".join(code.value for code in FindingCode if code in assess)
    return (
        f"This is a {VIEW_NAME[modality]}. Reply with JSON only, matching this schema: {SCHEMA}. "
        f"Judge only these findings: {codes}. "
        '"possible" means a region that may show the finding, given as boxes [x0, y0, x1, y1] in '
        "integers from 0 to 1000 (thousandths of the image width and height, origin top left); "
        '"not_seen" means none was noticed. '
        'If the image cannot be judged, set image_quality to "unusable", give the reason and '
        "return no findings."
    )


def fit_image(image: Image.Image, max_px: int) -> Image.Image:
    """The picture as the model sees it: longest side at most `max_px`, aspect kept."""
    fitted = image.copy()
    fitted.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    return fitted
