import random

import firebase_admin
import os
from dotenv import load_dotenv
load_dotenv()  # โหลดค่าจากไฟล์ .env ตอนรันบนเครื่อง (ไม่มีผลตอน deploy จริงที่ตั้ง env var ผ่าน hosting)
from flask import Flask, request, abort, render_template

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

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# =====================================
# SAVE RESULT
# =====================================

def save_result(user_id, text):

    db.collection("results").document(user_id).set({
        "result": text
    })


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
        emotion_map.get(emotion, ["ผมรับฟังอยู่นะครับ"])
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
        "none": [""]
    }
 
    return random.choice(mapping.get(problem, [""]))


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
 
    return random.choice(mapping.get(support, [""]))


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
 
    return random.choice(mapping.get(style, [""]))


# =====================================
# INDEX
# =====================================

@app.route("/")
def index():
    return render_template("index.html")


# =====================================
# RECEIVE FROM LIFF
# =====================================

@app.route("/send", methods=["POST"])
def send():

    data = request.json

    user_id = data.get("userId")
    text = str(data.get("score"))

    if not user_id:
        return "bad request", 400

    try:

        save_result(user_id, text)

        line_bot_api.push_message(
            user_id,
            TextSendMessage(
                text="📋 ผลแบบประเมินของคุณ\n\n" + text
            )
        )

        return "OK"

    except Exception as e:

        print("ERROR :", e)

        return "error", 500


# =====================================
# CALLBACK
# =====================================

@app.route("/callback", methods=["POST"])
def callback():

    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)

    except InvalidSignatureError:
        abort(400)

    return "OK"


# =====================================
# MEMORY
# =====================================

user_state = {}


# =====================================
# LINE MESSAGE
# =====================================

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):

    user_id = event.source.user_id
    text = event.message.text.strip()

    # =================================
    # CREATE MEMORY
    # =================================

    if user_id not in user_state:
        user_state[user_id] = {}

    # =================================
    # OPEN LIFF
    # =================================

    if text == "ประเมิน":

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
        self_care_tips = (
            "💚 คำแนะนำการดูแลตนเองเบื้องต้น:\n\n"
            "1. 💤 นอนหลับพักผ่อนให้เพียงพอ 6-8 ชั่วโมงต่อวัน\n"
            "2. 🏃‍♂️ ออกกำลังกายสม่ำเสมออย่างน้อย 30 นาทีต่อวัน\n"
            "3. 🗣️ พูดคุยระบายความรู้สึกกับคนที่ไว้ใจ\n"
            "4. 🎨 ทำกิจกรรมนันทนาการหรือสิ่งที่ตนเองชอบเพื่อผ่อนคลาย"
        )
        
        quick_reply = QuickReply(
            items=[
                QuickReplyButton(action=MessageAction(label="💬 อยากระบายความรู้สึก", text="อยากระบายความรู้สึก")),
                QuickReplyButton(action=MessageAction(label="😰 เครียด / กังวล", text="เครียด / กังวล")),
                QuickReplyButton(action=MessageAction(label="📝 ประเมินสุขภาพจิต", text="ประเมิน"))
            ]
        )
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=self_care_tips, quick_reply=quick_reply)
        )
        return

    # =================================
    # AI PREDICT
    # =================================

    result = predict_all(text)

    risk = result["risk"]
    emotion = result["emotion"]
    problem = result["problem"]
    support = result["support_need"]
    intent = result["intent"]
    style = result["conversation_style"]

    print(result)

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

        print("LOG ERROR:", e)

    # =================================
    # SAVE MEMORY
    # =================================

    user_state[user_id]["emotion"] = emotion
    user_state[user_id]["problem"] = problem

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

    if risk in high_risk or intent == "crisis_help":

        reply = (
            "🚨 ผมเป็นห่วงคุณมากนะครับ\n\n"
            "กรุณาติดต่อสายด่วนสุขภาพจิต 1323 "
            "หรือคนใกล้ตัวทันที\n\n"
            "คุณไม่จำเป็นต้องอยู่คนเดียว"
        )

    else:

        replies = []

        # emotion
        if result["emotion_conf"] >= 0.50:
            replies.append(emotion_reply(emotion))

        # problem
        p_reply = ""

        if result["problem_conf"] >= 0.50:

            p_reply = problem_reply(problem)

        if p_reply:
            replies.append(p_reply)

        # support
        s_reply = ""

        if result["support_conf"] >= 0.45:

            s_reply = support_reply(support)

        if s_reply:
            replies.append(s_reply)

        # style
        st_reply = ""

        if result["style_conf"] >= 0.50:

            st_reply = style_reply(style)

        if st_reply:
            replies.append(st_reply)

        # fallback
        if len(replies) <= 1:
            replies.append(reply_normal(text))

        reply = "\n\n".join(replies)

    # =================================
    # SEND WITH QUICK REPLIES
    # =================================

    quick_reply = QuickReply(
        items=[
            QuickReplyButton(action=MessageAction(label="💬 อยากระบายความรู้สึก", text="อยากระบายความรู้สึก")),
            QuickReplyButton(action=MessageAction(label="😰 เครียด / กังวล", text="เครียด / กังวล")),
            QuickReplyButton(action=MessageAction(label="💚 คำแนะนำดูแลตนเอง", text="คำแนะนำการดูแลตนเอง")),
            QuickReplyButton(action=MessageAction(label="📝 ประเมินอีกครั้ง", text="ประเมิน"))
        ]
    )

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply, quick_reply=quick_reply)
    )


# =====================================
# RUN
# =====================================

if __name__ == "__main__":
    app.run(port=5000, debug=True)