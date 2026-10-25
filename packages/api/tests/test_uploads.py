import io

import pytest
from PIL import Image

from dentabridge_api.testing import jpeg_bytes
from dentabridge_api.uploads import UploadRejected, sanitise_upload
from dentabridge_core import ErrorCode


def _exif_with_gps_and_rotation(rotation: int) -> Image.Exif:
    exif = Image.Exif()
    exif[0x0112] = rotation
    exif[0x8825] = {1: "N", 2: (50.0, 27.0, 0.0), 3: "E", 4: (30.0, 31.0, 0.0)}
    return exif


def test_exif_gps_removed() -> None:
    original = jpeg_bytes(exif=_exif_with_gps_and_rotation(1))
    assert Image.open(io.BytesIO(original)).getexif().get_ifd(0x8825), "the fixture has GPS"
    clean = sanitise_upload(original, "image/jpeg")
    reopened = Image.open(io.BytesIO(clean.data))
    assert len(reopened.getexif()) == 0
    assert b"Exif" not in clean.data[:64]


def test_orientation_applied() -> None:
    sideways = jpeg_bytes((200, 100), exif=_exif_with_gps_and_rotation(6))  # rotate 90 on display
    clean = sanitise_upload(sideways, "image/jpeg")
    assert (clean.width, clean.height) == (100, 200)


def test_long_side_is_bounded_and_the_hash_is_of_the_output() -> None:
    clean = sanitise_upload(jpeg_bytes((3200, 1600)), "image/jpeg")
    assert max(clean.width, clean.height) == 1536
    import hashlib

    assert clean.sha256 == hashlib.sha256(clean.data).hexdigest()


@pytest.mark.parametrize(
    ("data", "content_type", "status", "code"),
    [
        (b"not an image at all", "image/jpeg", 400, ErrorCode.IMAGE_UNREADABLE),
        (b"", "image/png", 400, ErrorCode.IMAGE_UNREADABLE),
        (jpeg_bytes(), "text/plain", 415, ErrorCode.UNSUPPORTED_MEDIA_TYPE),
        (jpeg_bytes(), "application/pdf", 415, ErrorCode.UNSUPPORTED_MEDIA_TYPE),
        (jpeg_bytes((300, 300), noise=True), "image/jpeg", 413, ErrorCode.PAYLOAD_TOO_LARGE),
    ],
)
def test_400_413_415(data: bytes, content_type: str, status: int, code: ErrorCode) -> None:
    with pytest.raises(UploadRejected) as raised:
        sanitise_upload(data, content_type, max_bytes=1000)
    assert (raised.value.status, raised.value.code) == (status, code)


def test_too_many_pixels_are_refused_before_decoding() -> None:
    with pytest.raises(UploadRejected) as raised:
        sanitise_upload(jpeg_bytes((400, 400)), "image/jpeg", max_pixels=100_000)
    assert raised.value.code == ErrorCode.PAYLOAD_TOO_LARGE


def test_png_and_webp_are_accepted() -> None:
    for fmt, mime in (("PNG", "image/png"), ("WEBP", "image/webp")):
        buffer = io.BytesIO()
        Image.new("RGB", (50, 40), (10, 200, 10)).save(buffer, fmt)
        assert sanitise_upload(buffer.getvalue(), mime).width == 50
