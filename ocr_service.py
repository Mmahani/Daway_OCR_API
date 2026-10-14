import os
import re
import cv2
import tempfile

from difflib import SequenceMatcher
from paddleocr import PaddleOCR


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


# قاعدة بيانات مؤقتة
# لاحقاً سيتم استبدالها ببيانات Laravel
drug_database = [
    {
        "brand": "Panadol",
        "brand_aliases": ["Panadol", "بانادول", "بنادول"],
        "active_ingredients": ["Paracetamol", "باراسيتامول"]
    },
    {
        "brand": "Brufen",
        "brand_aliases": ["Brufen", "بروفين"],
        "active_ingredients": ["Ibuprofen", "ايبوبروفين", "إيبوبروفين"]
    },
    {
        "brand": "Augmentin",
        "brand_aliases": [
            "Augmentin",
            "اوجمنتين",
            "أوجمنتين",
            "اوجمين"
        ],
        "active_ingredients": [
            "Amoxicillin",
            "Clavulanic",
            "اموكسيسيلين"
        ]
    },
    {
        "brand": "Voltaren",
        "brand_aliases": ["Voltaren", "فولتارين"],
        "active_ingredients": ["Diclofenac", "ديكلوفيناك"]
    },
    {
        "brand": "Cipralex",
        "brand_aliases": ["Cipralex", "سيبرالكس"],
        "active_ingredients": ["Escitalopram", "اسيتالوبرام"]
    },
    {
        "brand": "Zoloft",
        "brand_aliases": ["Zoloft", "زولوفت"],
        "active_ingredients": ["Sertraline", "سيرترالين"]
    },
    {
        "brand": "Amaryl",
        "brand_aliases": ["Amaryl", "اماريل", "أماريل"],
        "active_ingredients": ["Glimepiride", "جليميبرايد"]
    },
    {
        "brand": "Concor",
        "brand_aliases": ["Concor", "كونكور"],
        "active_ingredients": ["Bisoprolol", "بيسوبرولول"]
    },
    {
        "brand": "Glucophage",
        "brand_aliases": [
            "Glucophage",
            "جلوكوفاج",
            "غلوكوفاج"
        ],
        "active_ingredients": ["Metformin", "ميتفورمين"]
    }
]


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



def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    try:
        # تقليل أبعاد الصور الكبيرة لتخفيف استهلاك الذاكرة
        height, width = img.shape[:2]
        max_side = 1200

        scale = min(1.0, max_side / max(height, width))

        if scale < 1.0:
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))

            img = cv2.resize(
                img,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    if cv2.imread(image_path) is None:

        return {
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

    original_texts = read_medicine_image(
        image_path
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
        result = original_result

    else:

        enhanced_path = preprocess_image(
            image_path
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
                result = enhanced_result

            elif enhanced_result is None:
                result = original_result

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
                result = enhanced_result

            else:
                result = original_result

        finally:

            if (
                enhanced_path
                and os.path.exists(
                    enhanced_path
                )
            ):
                os.remove(enhanced_path)

    if result is None:

        return {
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

    return {
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
