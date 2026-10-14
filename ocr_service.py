import json
import logging
import os
import re
import cv2
import tempfile

from difflib import SequenceMatcher
from paddleocr import PaddleOCR


logger = logging.getLogger(__name__)

MAX_SIDE = 1600


ocr = None
ocr_error = None


def get_ocr():

    global ocr, ocr_error

    if ocr is None and ocr_error is None:

        try:

            ocr = PaddleOCR(
                lang="ar",
                ocr_version="PP-OCRv3",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )

        except Exception as error:
            ocr_error = error

    if ocr is None:

        raise RuntimeError(
            "تعذر تحميل نموذج OCR: "
            + str(ocr_error)
        )

    return ocr


def is_model_loaded():

    return ocr is not None


# قاعدة بيانات الأدوية
# تُحمّل من drugs.json عند بدء التشغيل
def load_drug_database():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "drugs.json"
    )

    with open(path, encoding="utf-8") as drugs_file:
        return json.load(drugs_file)


drug_database = load_drug_database()

logger.info(
    "Loaded %d drugs from drugs.json",
    len(drug_database)
)


def debug_enabled():

    return os.environ.get(
        "DEBUG_OCR",
        ""
    ).lower() == "true"


def get_ocr_data(result):
    data = result.json

    if callable(data):
        data = data()

    if "res" in data:
        data = data["res"]

    return data


def clean_ocr_text(text):
    text = str(text).strip()
    text = re.sub(r"[^\w\s\u0600-\u06FF\-]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def read_medicine_image(image_path):
    results = get_ocr().predict(image_path)

    detected_texts = []

    for result in results:
        data = get_ocr_data(result)

        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])

        for text, score in zip(texts, scores):
            cleaned = clean_ocr_text(text)

            if cleaned:
                detected_texts.append({
                    "text": cleaned,
                    "confidence": round(float(score), 3)
                })

    logger.info(
        "OCR detected texts: %s",
        [item["text"] for item in detected_texts]
    )

    return detected_texts


def normalize_for_match(text):
    text = str(text).lower().strip()

    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")

    text = re.sub(
        r"[^a-z\u0600-\u06FF\s]",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_top_matches(
    detected_texts,
    database,
    top_n=3
):
    matches = []

    for item in detected_texts:

        raw_text = item["text"]
        normalized_raw = normalize_for_match(raw_text)

        if not normalized_raw:
            continue

        raw_words = normalized_raw.split()

        for drug in database:

            for alias in drug["brand_aliases"]:

                normalized_alias = normalize_for_match(alias)

                scores = []

                full_score = SequenceMatcher(
                    None,
                    normalized_raw,
                    normalized_alias
                ).ratio()

                scores.append(full_score)

                for word in raw_words:

                    if word == normalized_alias:
                        word_score = 1.0

                    elif (
                        len(word) >= 4
                        and len(normalized_alias) >= 4
                        and abs(
                            len(word)
                            - len(normalized_alias)
                        ) <= 1
                    ):
                        word_score = SequenceMatcher(
                            None,
                            word,
                            normalized_alias
                        ).ratio()

                    else:
                        word_score = 0

                    scores.append(word_score)

                best_score = max(scores)

                matches.append({
                    "drug_name": drug["brand"],
                    "matched_alias": alias,
                    "ocr_text": raw_text,
                    "score": round(best_score, 3)
                })

    matches = sorted(
        matches,
        key=lambda x: x["score"],
        reverse=True
    )

    unique = []
    seen = set()

    for match in matches:

        if match["drug_name"] not in seen:
            unique.append(match)
            seen.add(match["drug_name"])

        if len(unique) == top_n:
            break

    return unique


def build_ocr_response(top_matches):

    if not top_matches:
        return {
            "ocr_success": False,
            "status": "not_found",
            "drug_name": None,
            "best_candidate": None,
            "match_score": 0,
            "message": "لم يتم التعرف على اسم الدواء"
        }

    best = top_matches[0]
    score = best["score"]

    if score >= 0.80:

        status = "confirmed"
        drug_name = best["drug_name"]

        message = (
            "تم التعرف على الدواء بنجاح"
        )

    elif score >= 0.65:

        status = "needs_confirmation"
        drug_name = None

        message = (
            "تم العثور على اسم محتمل "
            "ويحتاج إلى تأكيد"
        )

    else:

        status = "uncertain"
        drug_name = None

        message = (
            "لم يتم التعرف على اسم "
            "الدواء بثقة"
        )

    return {
        "ocr_success": True,
        "status": status,
        "drug_name": drug_name,
        "best_candidate": best["drug_name"],
        "raw_text": best["ocr_text"],
        "matched_alias": best["matched_alias"],
        "match_score": score,
        "message": message
    }



def downscale_if_needed(image_path):
    """Returns (path_to_use, temp_path_to_delete_or_None)."""

    img = cv2.imread(image_path)

    if img is None:
        return None, None

    height, width = img.shape[:2]

    if max(height, width) <= MAX_SIDE:
        return image_path, None

    scale = MAX_SIDE / max(height, width)

    resized = cv2.resize(
        img,
        (
            max(1, int(width * scale)),
            max(1, int(height * scale))
        ),
        interpolation=cv2.INTER_AREA
    )

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    temp_path = temp_file.name
    temp_file.close()

    success = cv2.imwrite(
        temp_path,
        resized,
        [cv2.IMWRITE_JPEG_QUALITY, 85]
    )

    if not success:
        logger.warning(
            "Downscale write failed, using original image"
        )
        os.remove(temp_path)

        return image_path, None

    logger.info(
        "Downscaled image %dx%d -> %s",
        width,
        height,
        temp_path
    )

    return temp_path, temp_path


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    try:
        height, width = img.shape[:2]

        scale = min(3, MAX_SIDE / max(height, width))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if scale != 1.0:
            gray = cv2.resize(
                gray,
                (
                    max(1, int(width * scale)),
                    max(1, int(height * scale))
                ),
                interpolation=(
                    cv2.INTER_CUBIC
                    if scale > 1.0
                    else cv2.INTER_AREA
                )
            )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        try:
            temp_path = temp_file.name
        finally:
            temp_file.close()

        success = cv2.imwrite(
            temp_path,
            enhanced,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if not success:
            os.remove(temp_path)
            return None

        return temp_path

    finally:
        del img


def process_medicine_image(image_path):

    ocr_path, downscaled_temp = downscale_if_needed(
        image_path
    )

    if ocr_path is None:

        response = {
            "source": "ocr",
            "intent": "search_medicine",
            "drug_name": None,
            "symptoms": [],
            "ocr_success": False,
            "status": "not_found",
            "best_candidate": None,
            "match_score": 0,
            "message": (
                "تعذر قراءة الصورة"
            )
        }

        if debug_enabled():
            response["raw_texts"] = []
            response["top_matches"] = []

        return response

    try:

        original_texts = read_medicine_image(
            ocr_path
        )

        if original_texts:

            original_matches = get_top_matches(
                original_texts,
                drug_database,
                top_n=3
            )

            original_result = build_ocr_response(
                original_matches
            )

        else:
            original_result = None

        # إذا النتيجة الأصلية مؤكدة
        # لا نحتاج لتحسين الصورة
        if (
            original_result
            and original_result.get("status")
            == "confirmed"
        ):
            winner = (
                original_result,
                original_texts,
                original_matches
                if original_texts
                else []
            )

        else:

            enhanced_path = preprocess_image(
                ocr_path
            )

            try:

                if enhanced_path:

                    enhanced_texts = (
                        read_medicine_image(
                            enhanced_path
                        )
                    )

                else:
                    enhanced_texts = []

                if enhanced_texts:

                    enhanced_matches = (
                        get_top_matches(
                            enhanced_texts,
                            drug_database,
                            top_n=3
                        )
                    )

                    enhanced_result = (
                        build_ocr_response(
                            enhanced_matches
                        )
                    )

                else:
                    enhanced_result = None

                if original_result is None:
                    winner = (
                        enhanced_result,
                        enhanced_texts,
                        enhanced_matches
                        if enhanced_texts
                        else []
                    )

                elif enhanced_result is None:
                    winner = (
                        original_result,
                        original_texts,
                        original_matches
                        if original_texts
                        else []
                    )

                elif (
                    enhanced_result.get(
                        "match_score",
                        0
                    )
                    >
                    original_result.get(
                        "match_score",
                        0
                    )
                ):
                    winner = (
                        enhanced_result,
                        enhanced_texts,
                        enhanced_matches
                        if enhanced_texts
                        else []
                    )

                else:
                    winner = (
                        original_result,
                        original_texts,
                        original_matches
                        if original_texts
                        else []
                    )

            finally:

                if (
                    enhanced_path
                    and os.path.exists(
                        enhanced_path
                    )
                ):
                    os.remove(enhanced_path)

    finally:

            if (
                downscaled_temp
                and os.path.exists(
                    downscaled_temp
                )
            ):
                os.remove(downscaled_temp)

    result, winning_texts, winning_matches = winner

    if result is None:

        response = {
            "source": "ocr",
            "intent": "search_medicine",
            "drug_name": None,
            "symptoms": [],
            "ocr_success": False,
            "status": "not_found",
            "best_candidate": None,
            "match_score": 0,
            "message": (
                "لم يتم العثور على "
                "نص واضح في الصورة"
            )
        }

        if debug_enabled():
            response["raw_texts"] = []
            response["top_matches"] = []

        logger.info(
            "OCR final: status=not_found"
        )

        return response

    response = {
        "source": "ocr",
        "intent": "search_medicine",
        "drug_name": result.get(
            "drug_name"
        ),
        "symptoms": [],
        "ocr_success": result.get(
            "ocr_success"
        ),
        "status": result.get(
            "status"
        ),
        "best_candidate": result.get(
            "best_candidate"
        ),
        "match_score": result.get(
            "match_score"
        ),
        "message": result.get(
            "message"
        )
    }

    if debug_enabled():
        response["raw_texts"] = winning_texts
        response["top_matches"] = winning_matches

    logger.info(
        "OCR final: status=%s drug=%s score=%s",
        response["status"],
        response["drug_name"],
        response["match_score"]
    )

    return response
