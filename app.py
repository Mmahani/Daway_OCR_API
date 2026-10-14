import asyncio
import os
import tempfile

import cv2
import numpy as np

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Header
)

from ocr_service import (
    process_medicine_image,
    is_model_loaded
)


MAX_IMAGE_BYTES = int(
    os.environ.get("MAX_IMAGE_MB", "10")
) * 1024 * 1024

OCR_TIMEOUT_SEC = float(
    os.environ.get("OCR_TIMEOUT_SEC", "120")
)

API_KEY = os.environ.get(
    "OCR_API_KEY",
    ""
)


def check_api_key(x_api_key: str | None):

    if API_KEY and x_api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="مفتاح API غير صالح"
        )


app = FastAPI(
    title="Daway Medicine OCR API",
    version="1.0"
)


@app.get("/")
def home():

    return {
        "status": "success",
        "message": "Daway OCR API is running"
    }


@app.get("/health")
def health():

    loaded = is_model_loaded()

    return {
        "status": "ok" if loaded else "starting",
        "model_loaded": loaded
    }


@app.post("/ocr/medicine")
async def medicine_ocr(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None)
):

    check_api_key(x_api_key)

    allowed_types = [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ]

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="يرجى رفع صورة JPG أو PNG"
        )

    suffix = os.path.splitext(
        file.filename
    )[1]

    if not suffix:
        suffix = ".jpg"

    temp_path = None

    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="الصورة فارغة"
            )

        if len(image_bytes) > MAX_IMAGE_BYTES:

            raise HTTPException(
                status_code=413,
                detail="حجم الصورة كبير، الحد الأقصى 10MB"
            )

        if cv2.imdecode(
            np.frombuffer(
                image_bytes,
                dtype=np.uint8
            ),
            cv2.IMREAD_COLOR
        ) is None:

            raise HTTPException(
                status_code=400,
                detail="الملف ليس صورة صالحة"
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(image_bytes)

            temp_path = temp_file.name

        result = await asyncio.wait_for(
            asyncio.to_thread(
                process_medicine_image,
                temp_path
            ),
            timeout=OCR_TIMEOUT_SEC
        )

        return result

    except asyncio.TimeoutError:

        raise HTTPException(
            status_code=504,
            detail="انتهت مهلة معالجة الصورة، حاول مجدداً"
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(temp_path)