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

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    TemplateSendMessage,
    ButtonsTemplate,
    URIAction,
    QuickReply,
    QuickReplyButton,
    MessageAction
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
            "ผมอยู่ตรงนี้นะ ถ้าอยากเล่าอะไร 😊",
            "มีอะไรอยากคุยเพิ่มเติมไหมครับ",
            "ผมพร้อมรับฟังคุณเสมอครับ"
        ])


# =====================================
# EMOTION RESPONSE
# =====================================

def emotion_reply(emotion):

    emotion_map = {

        "happy": [
            "😊 รู้สึกดีจังที่เห็นคุณมีความสุขนะ",
            "ยินดีด้วยกับความรู้สึกดีๆ วันนี้นะครับ 🌸",
            "เห็นคุณสดใสแบบนี้ ผมก็พลอยดีใจไปด้วยนะครับ 🌸"
        ],

        "sad": [
            "😔 ฟังดูเหมือนคุณกำลังเสียใจนะ",
            "ผมอยู่ตรงนี้นะ ลองเล่าให้ฟังได้",
            "ถ้าอยากร้องไห้ก็ร้องได้นะครับ ไม่ต้องฝืน"
        ],

        "angry": [
            "😣 ดูเหมือนคุณกำลังโมโหหรืออึดอัด",
            "ลองค่อยๆ เล่าให้ฟังได้นะ ผมรับฟังเสมอ",
            "อารมณ์แบบนี้เป็นเรื่องปกตินะครับ ค่อยๆ ระบายออกมาได้"
        ],

        "fear": [
            "😰 คุณรู้สึกกังวลหรือกลัวอะไรอยู่รึเปล่าครับ",
            "ค่อยๆ หายใจนะ มีเรื่องอะไรที่ทำให้กังวลใจเป็นพิเศษไหม",
            "ความกลัวเป็นเรื่องธรรมชาติ ลองบอกผมได้นะว่ากังวลเรื่องอะไร"
        ],

        "tired": [
            "🥱 วันนี้ดูเหนื่อยล้ามากเลยนะ หาเวลาพักผ่อนบ้างนะครับ",
            "เข้าใจเลยนะว่าช่วงนี้เหนื่อยมาก กอดปลอบใจนะครับ 💙",
            "ร่างกายกำลังบอกให้พักนะครับ อย่าฝืนตัวเองมากไป"
        ],

        "confused": [
            "🤔 ฟังดูเหมือนคุณกำลังสับสนหรือลังเลอยู่ใช่ไหมครับ",
            "ค่อยๆ คิดนะ ลองคุยกันดูก่อนได้ว่าสับสนเรื่องไหนอยู่",
            "ไม่ต้องรีบหาคำตอบตอนนี้ก็ได้นะครับ ค่อยๆ คิดไปทีละขั้น"
        ],

        "neutral": [
            "😊 ขอบคุณที่คุยกับผมนะ",
            "ผมพร้อมรับฟังและเคียงข้างคุณเสมอ",
            "มีอะไรอยากเล่าเพิ่มไหมครับ ผมพร้อมฟังเสมอ"
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
            "📚 เรื่องเรียน/การเรียนกดดันคุณอยู่ใช่ไหมครับ",
            "📚 ดูเหมือนเรื่องเรียนจะหนักใจคุณอยู่ไม่น้อยเลยนะครับ",
            "📚 การเรียนบางช่วงก็กดดันจริงๆ ค่อยๆ จัดการไปทีละเรื่องนะครับ"
        ],

        "work": [
            "💼 เรื่องงานอาจทำให้เหนื่อยมากเลยนะ",
            "💼 ฟังดูเหมือนงานตอนนี้หนักหนาเอาการเลยนะครับ",
            "💼 เรื่องงานเป็นอะไรที่กดดันได้ง่ายจริงๆ พักบ้างนะครับ"
        ],

        "relationship": [
            "💔 เรื่องความสัมพันธ์อาจทำให้เจ็บปวดมาก",
            "💔 เรื่องคนที่เรารักมันกระทบใจได้ลึกจริงๆ นะครับ",
            "💔 ความสัมพันธ์ที่มีปัญหาทำให้เหนื่อยใจได้มากเลย"
        ],

        "family": [
            "👨‍👩‍👧 ปัญหาครอบครัวส่งผลต่อความรู้สึกได้มากเลย",
            "👨‍👩‍👧 เรื่องในบ้านบางทีก็หนักใจที่สุดเลยนะครับ",
            "👨‍👩‍👧 ความสัมพันธ์ในครอบครัวเป็นเรื่องที่กระทบใจได้ลึกจริงๆ"
        ],

        "financial": [
            "💸 เรื่องเงินสามารถสร้างความเครียดได้จริงๆ",
            "💸 ปัญหาการเงินเป็นความเครียดที่กดดันได้ตลอดเวลาเลยนะครับ",
            "💸 เรื่องค่าใช้จ่ายทำให้กังวลใจได้มากจริงๆ"
        ],

        "social": [
            "👥 เรื่องคนรอบตัวหรือเพื่อนอาจทำให้อึดอัดได้",
            "👥 การเข้าสังคมบางทีก็เหนื่อยใจไม่น้อยเลยนะครับ",
            "👥 ความสัมพันธ์กับคนรอบตัวเป็นเรื่องที่กระทบใจได้มาก"
        ],

        "health": [
            "🩺 เรื่องปัญหาสุขภาพก็มีผลต่อสภาพจิตใจของเรานะ",
            "🩺 สุขภาพกายกับใจเชื่อมโยงกันเสมอ ดูแลตัวเองด้วยนะครับ",
            "🩺 เรื่องสุขภาพเป็นสิ่งที่กังวลใจได้ง่ายจริงๆ"
        ],

        "self_esteem": [
            "🌟 ความภูมิใจหรือความมั่นใจในตัวเองเป็นเรื่องสำคัญนะ ค่อยๆ เสริมความมั่นใจกันไป",
            "🌟 บางทีเราก็โหดร้ายกับตัวเองเกินไป ค่อยๆ ใจดีกับตัวเองบ้างนะครับ",
            "🌟 คุณค่าของคุณไม่ได้ขึ้นอยู่กับความรู้สึกแย่ๆ ตอนนี้เลยนะครับ"
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
            "ถ้าต้องการ ผมสามารถช่วยแนะนำวิธีรับมือเบื้องต้นได้นะ",
            "อยากให้ผมช่วยแนะนำแนวทางไหม ลองบอกรายละเอียดเพิ่มได้นะครับ",
            "ผมพอจะช่วยแนะนำวิธีจัดการเบื้องต้นได้ ถ้าอยากลองดูนะครับ"
        ],

        "encouragement": [
            "คุณเก่งมากแล้วที่พยายามผ่านมาได้ถึงตอนนี้ 💙",
            "แค่คุณยังพยายามอยู่ตอนนี้ก็เก่งมากแล้วนะครับ 💙",
            "คุณผ่านเรื่องยากๆ มาได้มากกว่าที่คิดเสมอนะครับ 💙"
        ],

        "calming": [
            "ลองทำใจให้สบาย ค่อยๆ หายใจเข้าลึกๆ ผ่อนคลายร่างกายดูนะ",
            "ลองหายใจเข้าลึกๆ ช้าๆ สัก 3 ครั้งดูนะครับ ค่อยๆ ผ่อนคลาย",
            "ใจเย็นๆ นะครับ ลองนั่งนิ่งๆ สักครู่ ค่อยๆ ปล่อยความตึงเครียดออกไป"
        ],

        "information": [
            "ถ้าอยากได้ข้อมูลความรู้เพิ่มเติมเกี่ยวกับสุขภาพจิต บอกผมได้เลยนะ",
            "มีอะไรอยากรู้เพิ่มเรื่องสุขภาพจิตไหมครับ ผมช่วยอธิบายได้",
            "ถ้าสงสัยเรื่องอาการหรือแนวทางดูแลใจ ถามผมได้เลยนะครับ"
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
            "มาคุยกันเรื่อยๆ แบบนี้นะครับ ผมอยู่ตรงนี้"
        ],

        "serious": [
            "ผมจะตั้งใจช่วยเหลือและรับมือเรื่องนี้อย่างเต็มที่ครับ",
            "เรื่องนี้สำคัญนะครับ ผมจะตั้งใจฟังและช่วยอย่างจริงจัง",
            "ผมรับรู้ว่าเรื่องนี้หนักใจ ผมจะช่วยดูแลอย่างเต็มที่ครับ"
        ],

        "direct": [
            "ตรงไปตรงมานะ ผมเข้าใจคุณครับ",
            "เข้าใจแล้วครับ ผมจะตอบตรงประเด็นให้เลยนะ",
            "รับทราบครับ ขอตอบแบบตรงๆ เลยนะครับ"
        ],

        "detailed": [
            "ผมจะช่วยลงรายละเอียดและวิเคราะห์ปัญหาไปพร้อมกับคุณนะ",
            "ผมจะค่อยๆ อธิบายให้ละเอียดขึ้นนะครับ",
            "ลองมาไล่ดูทีละประเด็นให้ชัดเจนกันนะครับ"
        ],

        "gentle": [
            "ไม่เป็นไรนะ ค่อยๆ ไปทีละอย่างด้วยกันอย่างอ่อนโยนนะ",
            "ค่อยๆ นะครับ ไม่ต้องรีบ ผมอยู่ตรงนี้เสมอ",
            "ใจเย็นๆ นะครับ เราค่อยๆ คุยกันไปทีละนิดได้"
        ],

        "motivational": [
            "เป็นกำลังใจให้นะครับ คุณก้าวข้ามผ่านเรื่องนี้ไปได้แน่นอน! 🌷",
            "สู้ๆ นะครับ ทุกก้าวที่คุณพยายามมันมีความหมายเสมอ 🌷",
            "คุณทำได้แน่นอนครับ ผมเป็นกำลังใจให้เต็มที่ 🌷"
        ],

        "friendly": [
            "เราพร้อมคุยเคียงข้างเป็นมิตรที่ดีกับคุณเสมอนะ 💙",
            "คุยกับผมได้เหมือนเพื่อนคนหนึ่งเลยนะครับ 💙",
            "ผมอยู่ตรงนี้เหมือนเพื่อนที่พร้อมรับฟังเสมอนะครับ 💙"
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

        quick_reply = QuickReply(

            items=[

                QuickReplyButton(
                    action=MessageAction(
                        label="💬 อยากระบายความรู้สึก",
                        text="อยากระบายความรู้สึก"
                    )
                ),

                QuickReplyButton(
                    action=MessageAction(
                        label="😰 เครียด / กังวล",
                        text="เครียด / กังวล"
                    )
                ),

                QuickReplyButton(
                    action=MessageAction(
                        label="📝 ประเมินสุขภาพจิต",
                        text="ประเมิน"
                    )
                )

            ]
        )

        line_bot_api.reply_message(

            event.reply_token,

            TextSendMessage(
                text=SELF_CARE_TIPS,
                quick_reply=quick_reply
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

    else:

        if pending_action:
            set_pending_action(user_id, None)

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

        if result["style_conf"] >= STYLE_THRESHOLD:

            replies.append(
                style_reply(style)
            )


        # ---------------------------------
        # Fallback
        # ---------------------------------

        if len(replies) <= 1:

            replies.append(
                reply_normal(text)
            )


        reply = "\n\n".join(
            replies
        )


    # =================================
    # SEND WITH QUICK REPLIES
    # =================================

    quick_reply = QuickReply(

        items=[

            QuickReplyButton(

                action=MessageAction(

                    label="💬 อยากระบายความรู้สึก",

                    text="อยากระบายความรู้สึก"

                )

            ),

            QuickReplyButton(

                action=MessageAction(

                    label="😰 เครียด / กังวล",

                    text="เครียด / กังวล"

                )

            ),

            QuickReplyButton(

                action=MessageAction(

                    label="💚 คำแนะนำดูแลตนเอง",

                    text="คำแนะนำการดูแลตนเอง"

                )

            ),

            QuickReplyButton(

                action=MessageAction(

                    label="📝 ประเมินอีกครั้ง",

                    text="ประเมิน"

                )

            )

        ]
    )


    line_bot_api.reply_message(

        event.reply_token,

        TextSendMessage(

            text=reply,

            quick_reply=quick_reply

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
