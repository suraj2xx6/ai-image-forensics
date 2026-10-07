from io import BytesIO

import pytest
from PIL import Image


@pytest.fixture
def jpeg_bytes():
    output = BytesIO()
    Image.new("RGB", (64, 48), (90, 120, 140)).save(output, format="JPEG", quality=88)
    return output.getvalue()


@pytest.fixture
def png_bytes():
    output = BytesIO()
    Image.new("RGBA", (48, 48), (120, 50, 200, 255)).save(output, format="PNG")
    return output.getvalue()
