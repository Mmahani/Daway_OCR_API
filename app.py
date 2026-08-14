import os
import tempfile

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from ocr_service import (
    process_medicine_image
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


@app.post("/ocr/medicine")
async def medicine_ocr(
    file: UploadFile = File(...)
):

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

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(image_bytes)

            temp_path = temp_file.name

        result = process_medicine_image(
            temp_path
        )

        return result

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