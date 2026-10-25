"""Turn an upload into a SanitisedImage in memory: orientation applied, no metadata, bounded size.

Nothing here touches the disk. Starlette would spool a multipart file larger than 1 MB to a
temporary file, so the spool size is raised above the largest accepted upload, and a body that
is larger than that is refused while it is being read.
"""

import hashlib
import io

from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.formparsers import MultiPartParser

from dentabridge_api.errors import ApiError
from dentabridge_core import ErrorCode, SanitisedImage

ALLOWED_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_UPLOAD_PIXELS = 40_000_000
LONG_SIDE = 1536
FORM_OVERHEAD = 256 * 1024  # the other form fields and the multipart framing

MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES + FORM_OVERHEAD * 2


class UploadRejected(ApiError):
    """An upload the API will not look at; carries the status and error code to answer with."""


def sanitise_upload(
    data: bytes,
    content_type: str,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_pixels: int = MAX_UPLOAD_PIXELS,
) -> SanitisedImage:
    if content_type.split(";")[0].strip().lower() not in ALLOWED_TYPES:
        raise UploadRejected(415, ErrorCode.UNSUPPORTED_MEDIA_TYPE)
    if len(data) > max_bytes:
        raise UploadRejected(413, ErrorCode.PAYLOAD_TOO_LARGE)
    try:
        with Image.open(io.BytesIO(data)) as opened:
            if opened.format not in ALLOWED_FORMATS:
                raise UploadRejected(415, ErrorCode.UNSUPPORTED_MEDIA_TYPE)
            if opened.width * opened.height > max_pixels:  # checked before any pixel is decoded
                raise UploadRejected(413, ErrorCode.PAYLOAD_TOO_LARGE)
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise UploadRejected(400, ErrorCode.IMAGE_UNREADABLE) from error
    image.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)  # no exif, no icc profile: nothing to carry along
    jpeg = buffer.getvalue()
    return SanitisedImage(
        data=jpeg, width=image.width, height=image.height, sha256=hashlib.sha256(jpeg).hexdigest()
    )
