import random
import re

import firebase_admin
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, json, jsonify, request, abort, render_template

from ai_model import predict_all
from responses import compose_response, select_phase1_route

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    TemplateSendMessage,
    ButtonsTemplate,
    URIAction
)

from firebase_admin import credentials, firestore


# =====================================
# Firebase
# =====================================

firebase_creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")

if firebase_creds_json:
    # กรณี deploy จริง
    cred_dict = json.loads(firebase_creds_json)
    cred = credentials.Certificate(cred_dict)
else:
    # กรณีรันในเครื่อง
    cred = credentials.Certificate("serviceAccountKey.json")

firebase_admin.initialize_app(cred)

db = firestore.client()


# =====================================
# Flask
# =====================================

app = Flask(__name__)


# =====================================
# LINE CONFIG
# =====================================

LINE_CHANNEL_ACCESS_TOKEN = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
LINE_CHANNEL_SECRET = os.environ["LINE_CHANNEL_SECRET"]
LINE_LOGIN_CHANNEL_ID = os.environ["LINE_LOGIN_CHANNEL_ID"]

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# =====================================
# SAVE RESULT
# =====================================

def save_result(user_id, text):

    db.collection("results").document(user_id).set({
        "result": text
    })


PENDING_ADVICE = "offer_coping_advice"
AFFIRMATIVE_REPLIES = {
    "ได้เลย", "โอเค", "ได้ครับ", "ได้ค่ะ", "เอาเลย", "ลองดู",
    "ครับ", "ค่ะ", "ตกลง", "ยินดี", "เอาครับ", "เอาค่ะ"
}
NEGATIVE_REPLIES = {
    "ไม่เป็นไร", "ไม่เอา", "ไม่ต้อง", "ไว้ก่อน", "ยังไม่อยาก",
    "ขอบคุณครับ", "ขอบคุณค่ะ"
}
SELF_CARE_TIPS = (
    "💚 คำแนะนำการดูแลตนเองเบื้องต้น:\n\n"
    "1. 💤 นอนหลับพักผ่อนให้เพียงพอ 6-8 ชั่วโมงต่อวัน\n"
    "2. 🏃‍♂️ ออกกำลังกายสม่ำเสมออย่างน้อย 30 นาทีต่อวัน\n"
    "3. 🗣️ พูดคุยระบายความรู้สึกกับคนที่ไว้ใจ\n"
    "4. 🎨 ทำกิจกรรมนันทนาการหรือสิ่งที่ตนเองชอบเพื่อผ่อนคลาย"
)


def get_pending_action(user_id):
    if not user_id:
        return None
    try:
        state = db.collection("conversation_state").document(user_id).get().to_dict() or {}
        return state.get("pending_action")
    except Exception as e:
        print("STATE READ ERROR:", type(e).__name__)
        return None


def set_pending_action(user_id, action):
    if not user_id:
        return
    try:
        state_ref = db.collection("conversation_state").document(user_id)
        if action is None:
            state_ref.delete()
        else:
            state_ref.set({"pending_action": action})
    except Exception as e:
        print("STATE WRITE ERROR:", type(e).__name__)


# =====================================
# NORMAL CHAT
# =====================================

def reply_normal(text):

    if "สวัสดี" in text or "หวัดดี" in text or "พูดคุย" in text.lower():

        return random.choice([
            "สวัสดีครับ 😊 วันนี้เป็นยังไงบ้าง",
            "หวัดดีครับ มีอะไรให้ช่วยไหม",
            "สวัสดีครับ อยากคุยเรื่องอะไรดี"
        ])

    elif "ขอบคุณ" in text:

        return "ยินดีมากครับ 💙"

    elif "เป็นยังไง" in text:

        return "ผมสบายดีครับ แล้วคุณล่ะ 😊"

    else:

        return random.choice([
            "ถ้าอยากเล่าต่อ ผมฟังอยู่นะครับ",
            "มีอะไรอยากคุยเพิ่มเติมไหมครับ",
            "เล่าต่อได้ตามสะดวกครับ ผมฟังอยู่"
        ])


# =====================================
# EMOTION RESPONSE
# =====================================

def emotion_reply(emotion):

    emotion_map = {

        "happy": [
            "ฟังดูว่ามีเรื่องดี ๆ เกิดขึ้นนะครับ",
            "ยินดีด้วยกับความรู้สึกดีๆ วันนี้นะครับ 🌸",
            "ฟังแล้วดูเป็นเรื่องที่คุณดีใจอยู่มากนะ"
        ],

        "sad": [
            "😔 ฟังดูเหมือนคุณกำลังเสียใจนะ",
            "ผมอยู่ตรงนี้นะ ลองเล่าให้ฟังได้",
            "ถ้าอยากพักหรือเล่าเรื่องนี้ต่อ ก็ทำได้ตามที่สบายใจนะครับ"
        ],

        "angry": [
            "😣 ดูเหมือนคุณกำลังโมโหหรืออึดอัด",
            "ลองค่อยๆ เล่าให้ฟังได้นะ ผมรับฟังเสมอ",
            "เรื่องนี้ทำให้รู้สึกอึดอัดได้ ถ้าอยากเล่าต่อ ผมฟังอยู่"
        ],

        "fear": [
            "เรื่องนี้ชวนให้กังวลอยู่เหมือนกันนะครับ",
            "ฟังดูว่าคุณกังวลอยู่ ถ้าอยากเล่าเพิ่มว่าอะไรหนักใจที่สุด ผมฟังครับ",
            "ความกลัวเป็นเรื่องธรรมชาติ ลองบอกผมได้นะว่ากังวลเรื่องอะไร"
        ],

        "tired": [
            "ฟังดูว่าวันนี้ใช้แรงไปเยอะเลยนะ",
            "ช่วงนี้ดูเหนื่อยอยู่มากนะ ถ้าอยากเล่า ผมฟังครับ",
            "ถ้าวันนี้รู้สึกล้า ก็ไม่จำเป็นต้องฝืนคุยต่อทันทีนะครับ"
        ],

        "confused": [
            "🤔 ฟังดูเหมือนคุณกำลังสับสนหรือลังเลอยู่ใช่ไหมครับ",
            "ค่อยๆ คิดนะ ลองคุยกันดูก่อนได้ว่าสับสนเรื่องไหนอยู่",
            "ไม่ต้องรีบหาคำตอบตอนนี้ก็ได้นะครับ ค่อยๆ คิดไปทีละขั้น"
        ],

        "neutral": [
            "😊 ขอบคุณที่คุยกับผมนะ",
            "ถ้าอยากเล่าเรื่องไหนต่อ ผมฟังอยู่ครับ",
            "อยากเล่าเพิ่มตรงไหนก็ได้ครับ"
        ]
    }

    return random.choice(
        emotion_map.get(
            emotion,
            ["ผมรับฟังอยู่นะครับ"]
        )
    )


# =====================================
# PROBLEM RESPONSE
# =====================================

def problem_reply(problem):

    mapping = {

        "academic": [
            "เรื่องเรียนตอนนี้ทำให้หนักใจอยู่หรือเปล่าครับ",
            "📚 ดูเหมือนเรื่องเรียนจะหนักใจคุณอยู่ไม่น้อยเลยนะครับ",
            "เรื่องเรียนหลายอย่างอาจทำให้เหนื่อย ค่อย ๆ เล่าได้ว่าอะไรค้างใจที่สุด"
        ],

        "work": [
            "💼 เรื่องงานอาจทำให้เหนื่อยมากเลยนะ",
            "งานช่วงนี้ดูหนักอยู่ไม่น้อยนะครับ",
            "💼 เรื่องงานเป็นอะไรที่กดดันได้ง่ายจริงๆ พักบ้างนะครับ"
        ],

        "relationship": [
            "💔 เรื่องความสัมพันธ์อาจทำให้เจ็บปวดมาก",
            "เรื่องความสัมพันธ์บางทีก็กระทบใจมาก",
            "💔 ความสัมพันธ์ที่มีปัญหาทำให้เหนื่อยใจได้มากเลย"
        ],

        "family": [
            "👨‍👩‍👧 ปัญหาครอบครัวส่งผลต่อความรู้สึกได้มากเลย",
            "เรื่องในบ้านตอนนี้ดูหนักใจอยู่",
            "👨‍👩‍👧 ความสัมพันธ์ในครอบครัวเป็นเรื่องที่กระทบใจได้ลึกจริงๆ"
        ],

        "financial": [
            "💸 เรื่องเงินสามารถสร้างความเครียดได้จริงๆ",
            "เรื่องเงินช่วงนี้อาจทำให้กังวลอยู่มาก",
            "💸 เรื่องค่าใช้จ่ายทำให้กังวลใจได้มากจริงๆ"
        ],

        "social": [
            "👥 เรื่องคนรอบตัวหรือเพื่อนอาจทำให้อึดอัดได้",
            "👥 การเข้าสังคมบางทีก็เหนื่อยใจไม่น้อยเลยนะครับ",
            "เรื่องกับคนรอบตัวบางทีก็กระทบความรู้สึกได้"
        ],

        "health": [
            "🩺 เรื่องปัญหาสุขภาพก็มีผลต่อสภาพจิตใจของเรานะ",
            "ถ้ากังวลเรื่องสุขภาพอยู่ ค่อย ๆ เล่าส่วนที่อยากพูดถึงได้ครับ",
            "🩺 เรื่องสุขภาพเป็นสิ่งที่กังวลใจได้ง่ายจริงๆ"
        ],

        "self_esteem": [
            "เรื่องนี้อาจกระทบความมั่นใจของคุณอยู่",
            "บางครั้งเราอาจเผลอตัดสินตัวเองแรงไปนะ",
            "ความรู้สึกแย่ตอนนี้ไม่ได้บอกทุกอย่างเกี่ยวกับตัวคุณนะ"
        ],

        "none": [
            ""
        ]
    }

    return random.choice(
        mapping.get(
            problem,
            [""]
        )
    )


# =====================================
# SUPPORT RESPONSE
# =====================================

def support_reply(support):

    mapping = {

        "listener": [
            "ผมพร้อมรับฟังคุณนะ ลองเล่าเพิ่มเติมได้เลย",
            "เล่าให้ผมฟังได้เลยนะครับ ผมอยู่ตรงนี้",
            "ผมตั้งใจฟังอยู่นะครับ ค่อยๆ เล่ามาได้เลย"
        ],

        "advice": [
            "ถ้าอยากลองคิดทางรับมือ ผมช่วยเริ่มจากเรื่องที่หนักใจที่สุดได้",
            "ถ้าอยากได้แนวทาง เราเริ่มจากเรื่องที่คุณอยากจัดการก่อนก็ได้",
            "ถ้าอยากลองคิดขั้นแรกด้วยกัน บอกได้ว่าติดอยู่ตรงไหน"
        ],

        "encouragement": [
            "คุณพยายามกับเรื่องนี้มาไม่น้อยเลยนะ",
            "การยังพยายามอยู่ในวันที่ยากก็มีความหมายแล้วนะ",
            "ที่ผ่านมาอาจมีหลายเรื่องที่คุณรับมือมาได้ ค่อย ๆ มองทีละเรื่องก็พอ"
        ],

        "calming": [
            "ถ้าสะดวก ลองพักสักครู่แล้วสังเกตลมหายใจของตัวเอง",
            "ถ้าอยากลอง ค่อย ๆ หายใจในจังหวะที่สบายของตัวเอง",
            "ตอนนี้ถ้าพอทำได้ ลองนั่งในที่ที่สบายแล้วพักสักครู่"
        ],

        "information": [
            "ถ้าอยากรู้เรื่องการดูแลใจด้านไหน บอกหัวข้อได้ครับ",
            "มีอะไรอยากรู้เพิ่มเรื่องสุขภาพจิตไหมครับ ผมช่วยอธิบายได้",
            "ถ้ามีคำถามเรื่องการดูแลใจ บอกประเด็นที่สงสัยได้ครับ"
        ],

        "crisis_support": [
            "หากรู้สึกไม่ไหวจริงๆ หรือวิกฤต สามารถติดต่อสายด่วน 1323 หรือคนใกล้ตัวได้ทันทีนะครับ",
            "ตอนนี้คุณไม่ต้องเผชิญเรื่องนี้คนเดียวนะครับ ติดต่อสายด่วนสุขภาพจิต 1323 ได้ตลอด 24 ชั่วโมง",
            "ถ้ารู้สึกอันตรายกับตัวเองตอนนี้ โทรสายด่วน 1323 หรือแจ้งคนใกล้ตัวด่วนที่สุดนะครับ"
        ]
    }

    return random.choice(
        mapping.get(
            support,
            [""]
        )
    )


# =====================================
# STYLE RESPONSE
# =====================================

def style_reply(style):

    mapping = {

        "casual": [
            "เราค่อยๆ คุยกันแบบเป็นกันเองได้นะ 😊",
            "คุยกันสบายๆ แบบนี้แหละครับ ไม่ต้องเกร็ง",
            "คุยกันเรื่อย ๆ ได้ครับ"
        ],

        "serious": [
            "เรื่องนี้สำคัญนะครับ ผมจะฟังให้ตรงกับสิ่งที่คุณเล่า",
            "เรื่องนี้สำคัญนะครับ ผมจะตั้งใจฟังและช่วยอย่างจริงจัง",
            "ฟังดูว่าเรื่องนี้หนักใจอยู่ ผมจะคุยอย่างระมัดระวังนะครับ"
        ],

        "direct": [
            "ขอตอบตรงประเด็นนะครับ",
            "เข้าใจแล้วครับ ผมจะตอบตรงประเด็นให้เลยนะ",
            "ได้ครับ ขอเข้าเรื่องที่คุณถามเลย"
        ],

        "detailed": [
            "ลองแยกเรื่องนี้เป็นประเด็นย่อย ๆ แล้วค่อยดูทีละส่วนกัน",
            "ผมจะค่อยๆ อธิบายให้ละเอียดขึ้นนะครับ",
            "ลองมาไล่ดูทีละประเด็นให้ชัดเจนกันนะครับ"
        ],

        "gentle": [
            "ค่อย ๆ คุยไปทีละเรื่องก็ได้ครับ",
            "ไม่ต้องรีบครับ เล่าเท่าที่สะดวก",
            "ใจเย็นๆ นะครับ เราค่อยๆ คุยกันไปทีละนิดได้"
        ],

        "motivational": [
            "เรื่องนี้อาจยาก แต่เริ่มจากส่วนที่พอไหวก่อนก็ได้ครับ",
            "ความพยายามเล็ก ๆ ของคุณวันนี้ก็นับนะครับ",
            "ยังไม่ต้องมั่นใจว่าจะทำได้ทั้งหมด แค่ก้าวแรกที่พอไหวก็พอ"
        ],

        "friendly": [
            "คุยกันแบบสบาย ๆ ได้ครับ",
            "ถ้าอยากเล่าอะไรต่อก็คุยได้ครับ",
            "ผมฟังอยู่นะครับ เล่าต่อได้ตามสะดวก"
        ]
    }

    return random.choice(
        mapping.get(
            style,
            [""]
        )
    )


# =====================================
# INDEX
# =====================================

@app.route("/")
def index():

    return render_template("index.html")


# =====================================
# RECEIVE FROM LIFF
# =====================================

def validate_assessment_answers(data):
    if not isinstance(data, dict) or set(data) != {"answers_2q", "answers_9q", "answers_8q"}:
        raise ValueError("Invalid assessment payload")

    answers_2q = data["answers_2q"]
    answers_9q = data["answers_9q"]
    answers_8q = data["answers_8q"]

    def valid_answers(answers, count, maximum):
        return (isinstance(answers, list) and len(answers) == count
                and all(type(answer) is int and 0 <= answer <= maximum for answer in answers))

    if not valid_answers(answers_2q, 2, 1):
        raise ValueError("Invalid 2Q answers")
    if not isinstance(answers_9q, list) or not isinstance(answers_8q, list):
        raise ValueError("Invalid assessment answers")

    if sum(answers_2q) == 0:
        if answers_9q or answers_8q:
            raise ValueError("Invalid 2Q flow")
    else:
        if not valid_answers(answers_9q, 9, 3):
            raise ValueError("Invalid 9Q answers")
        needs_8q = sum(answers_9q) >= 7 or answers_9q[8] >= 1
        if needs_8q:
            if not valid_answers(answers_8q, 8, 1):
                raise ValueError("Invalid 8Q answers")
        elif answers_8q:
            raise ValueError("Invalid 9Q flow")

    return answers_2q, answers_9q, answers_8q


def build_assessment_result(answers_2q, answers_9q, answers_8q):
    if sum(answers_2q) == 0:
        return "📋 ผลการประเมินสุขภาพจิตของคุณ\n\n• แบบประเมินคัดกรอง 2Q: ปกติ 😊"

    score_9q = sum(answers_9q)
    if score_9q < 7:
        text_9q = "ไม่มีภาวะซึมเศร้า หรือมีระดับน้อยมาก"
    elif score_9q <= 12:
        text_9q = "มีภาวะซึมเศร้า ระดับน้อย"
    elif score_9q <= 18:
        text_9q = "มีภาวะซึมเศร้า ระดับปานกลาง"
    else:
        text_9q = "มีภาวะซึมเศร้า ระดับรุนแรง"

    if not answers_8q:
        return (f"📋 ผลการประเมินสุขภาพจิตของคุณ\n\n• คะแนนรวม 9Q: {score_9q} / 27 คะแนน\n"
                f"• ผลการประเมิน: {text_9q}")

    weights_8q = (1, 2, 6, 8, 9, 4, 10, 4)
    score_8q = sum(weight for answer, weight in zip(answers_8q, weights_8q) if answer == 1)
    if score_8q == 0:
        text_8q = "ระดับน้อยมาก"
    elif score_8q <= 8:
        text_8q = "ระดับน้อย"
    elif score_8q <= 16:
        text_8q = "ระดับปานกลาง"
    else:
        text_8q = "ระดับรุนแรง"

    return (f"📋 ผลการประเมินสุขภาพจิตของคุณ\n\n"
            f"• คะแนนรวม 9Q: {score_9q} / 27 คะแนน ({text_9q})\n"
            f"• คะแนนความเสี่ยง 8Q: {score_8q} / 44 คะแนน ({text_8q})")


@app.route("/send", methods=["POST"])
def send():

    data = request.get_json(silent=True)

    try:
        answers = validate_assessment_answers(data)
    except ValueError:
        return jsonify(saved=False, pushed=False), 400

    text = build_assessment_result(*answers)

    authorization = request.headers.get("Authorization", "")
    scheme, separator, id_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not id_token.strip():
        return jsonify(saved=False, pushed=False), 401

    verify_request = Request(
        "https://api.line.me/oauth2/v2.1/verify",
        data=urlencode({
            "id_token": id_token.strip(),
            "client_id": LINE_LOGIN_CHANNEL_ID
        }).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST"
    )

    try:
        with urlopen(verify_request, timeout=5) as line_response:
            verified = json.loads(line_response.read())
    except HTTPError as e:
        status = 401 if e.code in (400, 401) else 503
        return jsonify(saved=False, pushed=False), status
    except (URLError, TimeoutError, ValueError):
        return jsonify(saved=False, pushed=False), 503

    if not isinstance(verified, dict) or not isinstance(verified.get("sub"), str) or not verified["sub"]:
        return jsonify(saved=False, pushed=False), 503

    user_id = verified["sub"]

    try:

        save_result(
            user_id,
            text
        )

    except Exception as e:

        print("SAVE ERROR :", type(e).__name__)

        return jsonify(saved=False, pushed=False), 500

    try:

        line_bot_api.push_message(
            user_id,
            TextSendMessage(
                text="📋 ผลแบบประเมินของคุณ\n\n" + text
            )
        )

        return jsonify(saved=True, pushed=True), 200

    except Exception as e:

        print("PUSH ERROR :", type(e).__name__)

        return jsonify(saved=True, pushed=False), 502


# =====================================
# CALLBACK
# =====================================

@app.route("/callback", methods=["POST"])
def callback():

    signature = request.headers.get(
        "X-Line-Signature"
    )
    if not signature or not signature.strip():
        abort(400)

    body = request.get_data(
        as_text=True
    )

    try:

        handler.handle(
            body,
            signature
        )

    except InvalidSignatureError:

        abort(400)

    return "OK"


def detect_explicit_safety(text: str) -> bool:
    text_lower = text.strip().lower()
    explicit_safety_phrases = [
        "อยากตาย",
        "ฆ่าตัวตาย",
        "คิดฆ่าตัวตาย",
        "คิดจะจบชีวิตตัวเอง",
        "อยากทำร้ายตัวเอง",
        "ทำร้ายตัวเอง",
        "ไม่อยากมีชีวิต",
        "ไม่อยากมีชีวิตอยู่",
        "ไม่อยากมีชีวิตอยู่แล้ว",
        "ไม่อยากอยู่แล้ว"
    ]
    if not any(phrase in text_lower for phrase in explicit_safety_phrases):
        return False

    # Only a complete, clear denial is excluded; mixed statements still trigger.
    compact_text = "".join(text_lower.split()).rstrip(".,!?！？。")
    clear_negation = re.fullmatch(
        r"(?:ฉัน|ผม|หนู)?(?:ตอนนี้)?(?:ไม่ได้|ไม่)"
        r"(?:อยากตาย|คิดฆ่าตัวตาย|อยากทำร้ายตัวเอง)(?:แล้ว)?",
        compact_text
    )
    return clear_negation is None


# =====================================
# LINE MESSAGE
# =====================================

@handler.add(
    MessageEvent,
    message=TextMessage
)
def handle_message(event):

    user_id = event.source.user_id
    text = event.message.text.strip()

    # =================================
    # OPEN LIFF
    # =================================

    if text == "ประเมิน":

        set_pending_action(user_id, None)

        template = ButtonsTemplate(

            title="แบบประเมินสุขภาพจิต",

            text="กดปุ่มเพื่อเริ่มทำแบบประเมิน",

            actions=[

                URIAction(
                    label="📝 เริ่มทำแบบประเมิน",
                    uri="https://liff.line.me/2009909099-QZqLLKsr"
                )

            ]
        )

        line_bot_api.reply_message(

            event.reply_token,

            TemplateSendMessage(

                alt_text="เริ่มทำแบบประเมิน",

                template=template
            )
        )

        return


    # =================================
    # SELF-CARE GUIDELINES IN CHAT
    # =================================

    elif text == "คำแนะนำการดูแลตนเอง":

        set_pending_action(user_id, None)

        line_bot_api.reply_message(

            event.reply_token,

            TextSendMessage(
                text=SELF_CARE_TIPS
            )
        )

        return


    # =================================
    # AI PREDICT
    # =================================

    pending_action = get_pending_action(user_id)
    result = predict_all(text)

    risk = result["risk"]
    emotion = result["emotion"]
    problem = result["problem"]
    support = result["support_need"]
    intent = result["intent"]
    style = result["conversation_style"]

    # =================================
    # SAVE LOG
    # =================================

    try:

        db.collection("chat_logs").add({

            "user_id": user_id,

            "text": text,

            "prediction": result

        })

    except Exception as e:

        print(
            "LOG ERROR:",
            type(e).__name__
        )


    # =================================
    # RISK CHECK
    # =================================

    high_risk = [

        "risk_self_harm",

        "risk_suicidal_ideation",

        "risk_suicide_plan",

        "risk_immediate_danger",

        "risk_passive"

    ]


    # =================================
    # CONFIDENCE THRESHOLDS
    # =================================

    RISK_THRESHOLD = 0.65

    INTENT_CRISIS_THRESHOLD = 0.65

    EMOTION_THRESHOLD = 0.60

    PROBLEM_THRESHOLD = 0.55

    SUPPORT_THRESHOLD = 0.55

    STYLE_THRESHOLD = 0.50


    # =================================
    # EXPLICIT SAFETY RULE
    # =================================
    #
    # กรณีข้อความมีความหมายด้านความปลอดภัย
    # อย่างชัดเจน ให้ Safety Rule มีสิทธิ์
    # trigger Crisis แม้ ML confidence จะต่ำ
    #
    # สำคัญ:
    # ไม่ใส่คำกว้างๆ เช่น
    # "อยู่คนเดียว"
    # "เหนื่อย"
    # "เศร้า"
    # "ไม่อยากคุย"
    # เพราะอาจทำให้เกิด False Positive
    #

    explicit_safety = detect_explicit_safety(text)


    # =================================
    # ML RISK CHECK
    # =================================

    risk_is_high = (

        risk in high_risk

        and result["risk_conf"] >= RISK_THRESHOLD

    )


    # =================================
    # ML INTENT CHECK
    # =================================

    intent_is_crisis = (

        intent == "crisis_help"

        and result["intent_conf"] >= INTENT_CRISIS_THRESHOLD

    )


    # =================================
    # FINAL SAFETY DECISION
    # =================================

    offered_advice = False
    ordinary_route = select_phase1_route(
        text,
        eligible_intent=intent if result["intent_conf"] >= SUPPORT_THRESHOLD else None,
        eligible_support=support if result["support_conf"] >= SUPPORT_THRESHOLD else None,
    )
    if (
        explicit_safety
        or risk_is_high
        or intent_is_crisis
    ):

        if pending_action:
            set_pending_action(user_id, None)

        reply = (

            "🚨 ผมเป็นห่วงคุณมากนะครับ\n\n"

            "หากตอนนี้คุณกำลังรู้สึกไม่ปลอดภัย "
            "หรือมีความคิดทำร้ายตัวเอง "
            "กรุณาติดต่อสายด่วนสุขภาพจิต 1323 "
            "หรือคนใกล้ตัวทันที\n\n"

            "คุณไม่จำเป็นต้องอยู่คนเดียว"
        )


    # =================================
    # NORMAL RESPONSE
    # =================================

    elif pending_action == PENDING_ADVICE and text in AFFIRMATIVE_REPLIES:

        set_pending_action(user_id, None)
        reply = SELF_CARE_TIPS

    elif pending_action == PENDING_ADVICE and text in NEGATIVE_REPLIES:

        set_pending_action(user_id, None)
        reply = reply_normal(text)

    elif ordinary_route == "direct_advice":

        if pending_action:
            set_pending_action(user_id, None)

        reply = compose_response(
            text,
            route=ordinary_route,
            emotion=emotion if result["emotion_conf"] >= EMOTION_THRESHOLD else None,
            problem=problem if result["problem_conf"] >= PROBLEM_THRESHOLD else None,
            style=style if result["style_conf"] >= STYLE_THRESHOLD else None,
        ).text

    else:

        had_pending_action = bool(pending_action)
        if pending_action:
            set_pending_action(user_id, None)

        # Keep pending, information, and crisis-support responses on their existing path.
        if not had_pending_action and intent != "information" and not (
            result["support_conf"] >= SUPPORT_THRESHOLD
            and support in ("information", "crisis_support")
        ):
            draft = compose_response(
                text,
                route=ordinary_route,
                emotion=emotion if result["emotion_conf"] >= EMOTION_THRESHOLD else None,
                problem=problem if result["problem_conf"] >= PROBLEM_THRESHOLD else None,
                style=style if result["style_conf"] >= STYLE_THRESHOLD else None,
            )
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=draft.text),
            )
            if draft.offer_pending_advice:
                set_pending_action(user_id, PENDING_ADVICE)
            return

        replies = []


        # ---------------------------------
        # Emotion
        # ---------------------------------

        if result["emotion_conf"] >= EMOTION_THRESHOLD:

            replies.append(
                emotion_reply(emotion)
            )


        # ---------------------------------
        # Problem
        # ---------------------------------

        if result["problem_conf"] >= PROBLEM_THRESHOLD:

            p_reply = problem_reply(problem)

            if p_reply:

                if replies and support != "information":
                    replies[0] = p_reply
                elif not replies:
                    replies.append(
                        p_reply
                    )


        # ---------------------------------
        # Support
        # ---------------------------------

        if result["support_conf"] >= SUPPORT_THRESHOLD:

            # ถ้า Support Model บอก crisis_support
            # แต่ Safety Layer ยังไม่เข้า Crisis
            # จะไม่ส่งข้อความ Crisis โดยตรง

            if support == "crisis_support":

                s_reply = (

                    "ถ้าคุณกำลังรู้สึกหนักใจมาก "
                    "คุณสามารถเล่าให้ผมฟังเพิ่มเติมได้นะครับ "
                    "ผมพร้อมรับฟัง"

                )

            else:

                if support == "information" and "ดูแลใจ" in text and "เครียด" in text:
                    s_reply = (
                        "ถ้ารู้สึกเครียด ลองพักจากสิ่งที่ทำอยู่สักครู่ "
                        "แล้วค่อยสังเกตว่าเรื่องไหนทำให้หนักใจที่สุดก็ได้ครับ"
                    )
                else:
                    s_reply = support_reply(
                        support
                    )


            if s_reply:

                if support == "advice":
                    offered_advice = True

                replies.append(
                    s_reply
                )


        # ---------------------------------
        # Conversation Style
        # ---------------------------------

        if result["style_conf"] >= STYLE_THRESHOLD and not replies:

            replies.append(
                style_reply(style)
            )


        # ---------------------------------
        # Fallback
        # ---------------------------------

        if len(replies) <= 1 and not (
            support == "information" and result["support_conf"] >= SUPPORT_THRESHOLD
        ):

            replies.append(
                reply_normal(text)
            )


        reply = "\n\n".join(
            replies
        )


    # =================================
    # SEND REPLY
    # =================================

    line_bot_api.reply_message(

        event.reply_token,

        TextSendMessage(

            text=reply

        )
    )

    if offered_advice:
        set_pending_action(user_id, PENDING_ADVICE)


# =====================================
# RUN
# =====================================

# if __name__ == "__main__":

#     app.run(
#         port=5000,
#         debug=True
#     )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
