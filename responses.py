"""Ordinary LINE chat responses. Safety and assessment routing live in app.py."""

from dataclasses import dataclass
from random import choice
from typing import Callable, Sequence


EMOTION_RESPONSES = {
    "angry": (
        "ฟังดูว่าเรื่องนี้ทำให้คุณหงุดหงิด",
        "เรื่องที่เกิดขึ้นอาจทำให้อึดอัดใจ",
        "ฟังดูว่าคุณไม่พอใจกับเรื่องนี้",
    ),
    "confused": (
        "ฟังดูว่ามีหลายอย่างให้คิด",
        "ฟังดูว่าคุณยังไม่แน่ใจว่าจะเริ่มตรงไหน",
        "เรื่องนี้อาจมีหลายมุมที่ทำให้ลังเล",
    ),
    "fear": (
        "ฟังดูว่าคุณกำลังกังวลกับเรื่องนี้",
        "เรื่องนี้ทำให้รู้สึกกลัวได้",
        "ความไม่แน่นอนตรงนี้อาจชวนให้กังวล",
    ),
    "happy": (
        "ดีใจที่ได้ยินว่าคุณรู้สึกดีกับเรื่องนี้",
        "ฟังดูเป็นช่วงเวลาที่น่ายินดี",
        "เรื่องนี้ดูมีความหมายกับคุณมาก",
    ),
    "neutral": (
        "ขอบคุณที่เล่าให้ฟัง",
        "ได้ยินประเด็นที่คุณยกมาครับ",
        "ขอบคุณที่บอกสิ่งที่คุณกำลังคิด",
    ),
    "sad": (
        "ฟังดูว่าเรื่องนี้ทำให้คุณเสียใจ",
        "การเจอเรื่องแบบนี้อาจทำให้เศร้า",
        "ฟังดูว่าเรื่องนี้กระทบความรู้สึกคุณ",
    ),
    "tired": (
        "ฟังดูว่าคุณใช้พลังกับเรื่องนี้มาก",
        "ช่วงนี้คุณอาจรู้สึกอ่อนล้า",
        "เรื่องที่เจออาจทำให้หมดแรง",
    ),
}

PROBLEM_CONTEXTS = {
    "academic": (
        "เรื่องเรียนอาจมีหลายอย่างให้จัดการ",
        "ภาระการเรียนอาจกดดัน",
        "งานเรียนที่ค้างอยู่อาจทำให้เริ่มต้นยาก",
    ),
    "work": (
        "เรื่องงานอาจใช้พลังไปมาก",
        "ภาระงานอาจทำให้จัดลำดับยาก",
        "เรื่องในที่ทำงานอาจมีหลายอย่างให้คิด",
    ),
    "financial": (
        "เรื่องค่าใช้จ่ายอาจต้องค่อยจัดภาพรวมก่อน",
        "เรื่องเงินอาจทำให้ตัดสินใจลำบาก",
        "ภาระทางการเงินอาจกดดันหลายด้าน",
    ),
    "family": (
        "เรื่องในครอบครัวอาจกระทบความรู้สึก",
        "เรื่องที่บ้านอาจทำให้สื่อสารกันยาก",
        "ความสัมพันธ์ในบ้านอาจมีหลายมุมให้คำนึงถึง",
    ),
    "relationship": (
        "เรื่องความสัมพันธ์นี้อาจทำให้คิดหลายอย่าง",
        "การคุยกับคนสำคัญอาจไม่ง่ายเสมอไป",
        "เรื่องระหว่างกันอาจกระทบใจ",
    ),
    "social": (
        "การอยู่กับคนรอบตัวอาจมีแรงกดดัน",
        "เรื่องกับคนรอบข้างอาจทำให้อึดอัด",
        "ความสัมพันธ์ทางสังคมอาจมีหลายเรื่องให้จัดการ",
    ),
    "self_esteem": (
        "เรื่องนี้อาจทำให้คุณมองตัวเองเข้มงวดขึ้น",
        "ความมั่นใจอาจสั่นคลอนจากเรื่องนี้",
        "การคิดถึงตัวเองในช่วงนี้อาจไม่ง่าย",
    ),
    "health": (
        "เรื่องสุขภาพทำให้เกิดคำถามได้หลายอย่าง",
        "ความไม่แน่ใจด้านสุขภาพอาจทำให้กังวล",
        "หากข้อมูลเรื่องสุขภาพยังไม่ชัด อาจทำให้คิดมากได้",
    ),
    "none": (),
}

SUPPORT_RESPONSES = {
    "listener": (
        "เล่าเฉพาะส่วนที่อยากเล่าก็ได้ครับ",
        "ถ้าสะดวก เริ่มจากประเด็นที่อยากพูดถึงก่อนก็ได้",
        "ถ้าอยากระบายต่อ บอกได้ว่าเรื่องไหนสำคัญที่สุดตอนนี้",
    ),
    "advice": (
        "ถ้าต้องการ เราลองคิดทางเลือกเบื้องต้นกันได้",
        "หากอยากได้แนวทาง ลองเลือกเรื่องที่อยากเริ่มก่อน",
        "ถ้าอยากให้ช่วยคิดขั้นแรก บอกได้ว่าติดตรงไหน",
    ),
    "calming": (
        "ถ้าสะดวก ลองพักสักครู่แล้วสังเกตลมหายใจของตัวเอง",
        "หากอยากลอง ลองวางเท้าบนพื้นและสังเกตสิ่งรอบตัว",
        "ถ้าสะดวก ลองพักจากสิ่งที่กระตุ้นความกังวลชั่วครู่",
    ),
    "encouragement": (
        "คุณเลือกได้ว่าจะเริ่มจากส่วนที่ทำไหวก่อน",
        "คุณสามารถเริ่มจากขั้นเล็กที่ทำไหววันนี้",
        "เรื่องนี้ยากได้ และคุณไม่จำเป็นต้องแก้ทุกอย่างในครั้งเดียว",
    ),
}

DIRECT_ADVICE_BY_PROBLEM = {
    "academic": "ลองจดงานและกำหนดส่ง แล้วเลือกชิ้นที่ใกล้ที่สุดเพื่อเริ่มช่วงสั้น ๆ",
    "work": "ลองแยกงานที่ต้องทำวันนี้จากงานที่รอได้ แล้วเลือกเรื่องที่สำคัญที่สุดหนึ่งเรื่อง",
    "financial": "ลองรวบรวมรายรับ รายจ่ายจำเป็น และกำหนดชำระไว้ในรายการเดียวเพื่อเห็นภาพก่อนตัดสินใจ",
    "family": "ลองเลือกประเด็นเดียวที่อยากสื่อสาร และหาจังหวะที่ทั้งสองฝ่ายพร้อมคุย",
    "relationship": "ลองเขียนสิ่งที่อยากบอกและสิ่งที่อยากถามก่อนเริ่มคุย",
    "social": "ลองเลือกสถานการณ์ที่อยากคุยกับอีกฝ่าย และกำหนดขอบเขตที่อยากสื่อสารหนึ่งเรื่อง",
    "self_esteem": "ลองแยกสิ่งที่เกิดขึ้นจริงออกจากคำตัดสินต่อตัวเอง แล้วเลือกสิ่งที่ควบคุมได้หนึ่งข้อ",
    "health": "ลองจดอาการหรือคำถามที่กังวลไว้เพื่อปรึกษาผู้เชี่ยวชาญที่เหมาะสม",
    "none": "ลองระบุสิ่งที่อยากเปลี่ยนหรือแก้ก่อนหนึ่งเรื่อง แล้วเลือกก้าวเล็กที่ทำได้",
}

VENTING = (
    "ฟังดูว่าเรื่องนี้กระทบความรู้สึกคุณ ถ้าอยากเล่าต่อ เลือกส่วนที่สำคัญที่สุดได้ครับ",
    "เรื่องนี้อาจมีหลายอย่างปนกันอยู่ เล่าเฉพาะส่วนที่อยากพูดถึงก่อนได้",
    "ขอบคุณที่บอกเรื่องนี้ ถ้าอยากระบายต่อ ผมจะตามประเด็นที่คุณเลือก",
)

EMOTIONAL_SUPPORT = (
    "ฟังดูว่าวันนี้ไม่ง่ายสำหรับคุณ ลองเลือกสิ่งเล็ก ๆ ที่อยากให้ตัวเองได้พักจากเรื่องนี้",
    "ขอบคุณที่บอกความรู้สึกนี้ คุณเลือกได้ว่าจะเล่าต่อหรือพักการคุยไว้ก่อน",
    "เรื่องนี้อาจใช้พลังมาก ลองเริ่มจากสิ่งที่คุณรับมือไหวในตอนนี้",
)

SMALLTALK = {
    "greeting": "สวัสดีครับ วันนี้อยากคุยเรื่องอะไรดี",
    "thanks": "ยินดีครับ",
    "open_chat": "คุยกันได้ครับ มีเรื่องไหนที่อยากเริ่มไหม",
}

FALLBACK = (
    "ถ้าอยากคุยต่อ บอกได้ว่าอยากเริ่มที่เรื่องไหน",
    "ผมยังไม่แน่ใจว่าคุณอยากให้ช่วยแบบไหน เล่าเพิ่มอีกนิดได้ไหมครับ",
    "ถ้าอยากได้คำแนะนำหรือแค่อยากระบาย บอกแบบที่สะดวกได้ครับ",
)

STYLE_RULES = {
    "casual": "neutral",
    "detailed": "include_distinct_context",
    "direct": "action_first",
    "friendly": "neutral",
    "gentle": "neutral",
    "motivational": "neutral",
    "serious": "neutral",
}

DIRECT_ADVICE_PHRASES = (
    "ช่วยแนะนำหน่อย", "ขอคำแนะนำหน่อย", "แนะนำหน่อย",
    "ควรทำยังไง", "มีวิธีรับมือไหม", "ช่วยบอกวิธีหน่อย",
    "ช่วยแนะนำวิธี",
)


@dataclass(frozen=True)
class Fragment:
    text: str
    semantic_tag: str
    overlap_key: str | None = None


@dataclass(frozen=True)
class ResponseDraft:
    text: str
    route: str
    offer_pending_advice: bool = False
    semantic_tags: tuple[str, ...] = ()


Chooser = Callable[[Sequence[str]], str]


def get_emotion_ack(emotion: str | None, *, choose: Chooser = choice) -> Fragment | None:
    variants = EMOTION_RESPONSES.get(emotion, ())
    return Fragment(choose(variants), "acknowledgement", "feeling") if variants else None


def get_problem_context(problem: str | None, *, choose: Chooser = choice) -> Fragment | None:
    variants = PROBLEM_CONTEXTS.get(problem, ())
    return Fragment(choose(variants), "context", "topic") if variants else None


def get_support_action(route: str, *, choose: Chooser = choice) -> Fragment | None:
    variants = SUPPORT_RESPONSES.get(route, ())
    if not variants:
        return None
    tag = "invitation" if route in ("listener", "advice") else "action"
    return Fragment(choose(variants), tag, "tell_more" if tag == "invitation" else route)


def get_direct_advice(problem: str | None) -> Fragment:
    return Fragment(
        DIRECT_ADVICE_BY_PROBLEM.get(problem, DIRECT_ADVICE_BY_PROBLEM["none"]),
        "action",
        "first_step",
    )


def select_phase1_route(
    text: str, *, eligible_intent: str | None, eligible_support: str | None
) -> str:
    if any(phrase in text for phrase in ("แค่อยากให้ฟัง", "อยากให้ฟัง", "ยังไม่ต้องแนะนำ")):
        return "listener"
    if "อยากระบาย" in text or "ขอระบาย" in text:
        return "venting"
    if any(phrase in text for phrase in DIRECT_ADVICE_PHRASES):
        return "direct_advice"
    if eligible_intent == "ask_advice":
        return "direct_advice"
    if eligible_intent in ("venting", "emotional_support", "smalltalk"):
        return eligible_intent
    if eligible_support in SUPPORT_RESPONSES:
        return "advice_offer" if eligible_support == "advice" else eligible_support
    return "normal"


def _combine(fragments: list[Fragment], *, style: str | None) -> ResponseDraft:
    # Slot and overlap limits are checked before joining. Style changes order, not copy.
    selected: list[Fragment] = []
    tags: set[str] = set()
    overlaps: set[str] = set()
    for fragment in fragments:
        if fragment.semantic_tag in tags or (fragment.overlap_key and fragment.overlap_key in overlaps):
            continue
        selected.append(fragment)
        tags.add(fragment.semantic_tag)
        if fragment.overlap_key:
            overlaps.add(fragment.overlap_key)
    if STYLE_RULES.get(style) == "action_first":
        selected.sort(key=lambda item: item.semantic_tag != "action")
    return ResponseDraft(
        " ".join(item.text for item in selected),
        "normal",
        semantic_tags=tuple(item.semantic_tag for item in selected),
    )


def compose_response(
    text: str, *, route: str, emotion: str | None = None,
    problem: str | None = None, style: str | None = None,
    choose: Chooser = choice,
) -> ResponseDraft:
    if route == "direct_advice":
        # A grounded emotion may precede the action; no second offer to advise.
        fragments = [get_direct_advice(problem)]
        acknowledgement = get_emotion_ack(emotion, choose=choose) if emotion != "neutral" else None
        if acknowledgement:
            fragments.insert(0, acknowledgement)
        draft = _combine(fragments, style=style)
        return ResponseDraft(draft.text, route, semantic_tags=draft.semantic_tags)

    if route == "venting":
        return ResponseDraft(choose(VENTING), route, semantic_tags=("acknowledgement", "invitation"))
    if route == "emotional_support":
        return ResponseDraft(choose(EMOTIONAL_SUPPORT), route, semantic_tags=("acknowledgement", "action"))
    if route == "smalltalk":
        key = "thanks" if "ขอบคุณ" in text else "greeting" if ("สวัสดี" in text or "หวัดดี" in text) else "open_chat"
        return ResponseDraft(SMALLTALK[key], route, semantic_tags=("invitation",) if key != "thanks" else ())

    support_route = "advice" if route == "advice_offer" else route
    action = get_support_action(support_route, choose=choose)
    acknowledgement = get_emotion_ack(emotion, choose=choose) if emotion != "neutral" else None
    context = get_problem_context(problem, choose=choose)
    fragments: list[Fragment] = []
    if route == "calming":
        fragments.extend(item for item in (acknowledgement, context if not acknowledgement else None) if item)
    elif route in ("listener", "encouragement", "advice_offer"):
        fragments.extend(item for item in (context or acknowledgement,) if item)
    else:
        fragments.extend(item for item in (acknowledgement or context,) if item)
    if STYLE_RULES.get(style) == "include_distinct_context" and acknowledgement and context and acknowledgement not in fragments and context in fragments:
        fragments.insert(0, acknowledgement)
    elif STYLE_RULES.get(style) == "include_distinct_context" and acknowledgement and context and context not in fragments and acknowledgement in fragments:
        fragments.append(context)
    if action:
        fragments.append(action)
    if not fragments:
        return ResponseDraft(choose(FALLBACK), "fallback", semantic_tags=("invitation",))
    draft = _combine(fragments, style=style)
    return ResponseDraft(
        draft.text, route,
        offer_pending_advice=route == "advice_offer" and action is not None,
        semantic_tags=draft.semantic_tags,
    )
