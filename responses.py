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
        "ฟังดูแล้วเรื่องนี้ชวนให้กังวลอยู่เหมือนกันนะ",
        "ความไม่แน่นอนตรงนี้อาจชวนให้กังวล",
    ),
    "happy": (
        "ดีใจที่ได้ยินว่าคุณรู้สึกดีกับเรื่องนี้",
        "ฟังดูเป็นช่วงเวลาที่น่ายินดี",
        "ดูเป็นเรื่องที่มีความหมายกับคุณมากนะ",
    ),
    "neutral": (
        "ขอบคุณที่เล่าให้ฟัง",
        "ได้ครับ ค่อย ๆ เล่าต่อได้ตามที่สบายใจนะ",
        "ขอบคุณที่บอกสิ่งที่คุณกำลังคิด",
    ),
    "sad": (
        "ฟังดูว่าเรื่องนี้ทำให้คุณเสียใจ",
        "เจอเรื่องแบบนี้แล้วรู้สึกเสียใจได้เหมือนกันนะ",
        "ฟังดูว่าเรื่องนี้กระทบความรู้สึกคุณ",
    ),
    "tired": (
        "ฟังดูว่าคุณใช้พลังกับเรื่องนี้มาก",
        "ฟังดูว่าช่วงนี้คุณเหนื่อยอยู่มากเลยนะ",
        "เรื่องที่เจออาจทำให้หมดแรง",
    ),
}

PROBLEM_CONTEXTS = {
    "academic": (
        "เรื่องเรียนช่วงนี้ดูจะทำให้คุณต้องรับมือกับหลายอย่างเลยนะ",
        "ภาระการเรียนอาจกดดัน",
        "งานเรียนที่ค้างอยู่อาจทำให้เริ่มต้นยาก",
    ),
    "work": (
        "เรื่องงานอาจใช้พลังไปมาก",
        "ภาระงานอาจทำให้จัดลำดับยาก",
        "เรื่องในที่ทำงานอาจมีหลายอย่างให้คิด",
    ),
    "financial": (
        "เรื่องค่าใช้จ่ายช่วงนี้คงทำให้ต้องคิดหลายอย่างอยู่เหมือนกันนะ",
        "เรื่องเงินอาจทำให้ตัดสินใจลำบาก",
        "ภาระทางการเงินอาจกดดันหลายด้าน",
    ),
    "family": (
        "เรื่องในครอบครัวอาจกระทบความรู้สึก",
        "เรื่องที่บ้านอาจทำให้สื่อสารกันยาก",
        "เรื่องในบ้านบางทีก็กระทบความรู้สึกเราได้มากจริง ๆ",
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
        "เรื่องนี้อาจทำให้คุณตั้งคำถามกับตัวเองมากขึ้น",
        "ความมั่นใจอาจสั่นคลอนจากเรื่องนี้",
        "การคิดถึงตัวเองในช่วงนี้อาจไม่ง่าย",
    ),
    "health": (
        "เรื่องสุขภาพทำให้เกิดคำถามได้หลายอย่าง",
        "ความไม่แน่ใจด้านสุขภาพอาจทำให้กังวล",
        "ความไม่แน่ใจเรื่องสุขภาพแบบนี้ทำให้กังวลได้เหมือนกันนะ",
    ),
    "none": (),
}

SUPPORT_RESPONSES = {
    "listener": (
        "เล่าเท่าที่สบายใจได้เลยครับ ผมฟังอยู่",
        "ไม่ต้องเรียบเรียงให้ครบก็ได้ครับ ค่อย ๆ เล่าในส่วนที่อยากเล่าก็พอ",
        "ถ้ามีเรื่องไหนที่ยังค้างอยู่ในใจ เล่าให้ฟังได้ครับ ไม่ต้องรีบ",
    ),
    "advice": (
        "ถ้าต้องการ เราลองคิดทางเลือกเบื้องต้นกันได้",
        "ถ้าอยากลองหาทางออก เราเริ่มจากเรื่องที่ติดอยู่ที่สุดก่อนได้ครับ",
        "ถ้าอยากให้ช่วยคิดขั้นแรก บอกได้ว่าติดตรงไหน",
    ),
    "calming": (
        "งั้นลองค่อย ๆ หายใจเข้า–ออกช้า ๆ ดูนะครับ อาจช่วยให้รู้สึกผ่อนคลายขึ้นได้บ้าง",
        "ลองพักสักครู่ แล้วค่อย ๆ มองสิ่งรอบตัวทีละอย่างนะครับ อาจช่วยให้รู้สึกตั้งหลักได้มากขึ้น",
        "ยังไม่ต้องรีบจัดการทุกอย่างตอนนี้ก็ได้นะครับ ลองพักสักครู่ก่อน แล้วค่อยกลับมาดูทีละเรื่องก็ได้",
    ),
    "encouragement": (
        "ค่อย ๆ ทำทีละเรื่องก็พอนะครับ ตอนนี้ยังไม่จำเป็นต้องมั่นใจกับทุกอย่างก็ได้",
        "ยังไม่ต้องรีบทำทุกอย่างให้ได้ในครั้งเดียวครับ เริ่มจากสิ่งเล็ก ๆ ที่พอไหวก่อนก็ได้",
        "ถึงตอนนี้บางอย่างอาจยังไม่เป็นอย่างที่หวัง ก็ไม่ได้แปลว่าคุณทำได้ไม่ดีนะครับ ค่อย ๆ ให้เวลากับตัวเองแล้วไปทีละขั้นก็พอ",
    ),
}

DIRECT_ADVICE_BY_PROBLEM = {
    "academic": "ถ้าเรื่องเรียนตอนนี้มีหลายอย่างเข้ามาพร้อมกัน ลองจดงานที่ต้องทำออกมาก่อน แล้วค่อยเลือกเริ่มจากงานที่ใกล้กำหนดที่สุดก็ได้ครับ",
    "work": "ถ้างานตอนนี้มีหลายอย่างจนไม่รู้จะเริ่มตรงไหน ลองแยกก่อนว่าอะไรต้องทำวันนี้ แล้วค่อยเริ่มจากเรื่องที่สำคัญที่สุดสักหนึ่งอย่างครับ",
    "financial": "ถ้าช่วงนี้เรื่องค่าใช้จ่ายทำให้กังวล ลองเขียนรายรับกับค่าใช้จ่ายที่จำเป็นออกมาก่อน จะได้ค่อย ๆ ดูว่าเรื่องไหนควรจัดการก่อนครับ",
    "family": "ถ้าเป็นเรื่องในครอบครัว ลองเลือกคุยทีละประเด็นก่อนก็ได้ครับ แล้วหาจังหวะที่ทั้งสองฝ่ายพร้อมคุยกันมากที่สุด",
    "relationship": "ถ้ายังไม่แน่ใจว่าจะเริ่มคุยกับอีกฝ่ายยังไง ลองเรียบเรียงสิ่งที่อยากบอกไว้ก่อนสักนิดก็ได้ครับ จะได้พูดสิ่งที่สำคัญกับคุณได้ชัดขึ้น",
    "social": "ถ้าเป็นเรื่องกับคนรอบตัว ลองคิดก่อนว่ามีเรื่องไหนที่อยากให้อีกฝ่ายเข้าใจมากที่สุด แล้วค่อยเริ่มคุยจากเรื่องนั้นครับ",
    "self_esteem": "ถ้าตอนนี้กำลังรู้สึกว่าตัวเองทำได้ไม่ดีพอ ลองแยกก่อนว่าอะไรคือสิ่งที่เกิดขึ้นจริง กับอะไรที่เป็นคำตัดสินตัวเอง แล้วค่อยดูว่าส่วนไหนที่พอแก้ได้ครับ",
    "health": "ถ้ามีเรื่องสุขภาพที่ทำให้กังวล ลองจดอาการหรือสิ่งที่สงสัยไว้ก่อนนะครับ แล้วค่อยนำไปปรึกษาผู้เชี่ยวชาญที่เหมาะสม",
    "none": "ถ้ายังไม่รู้ว่าจะเริ่มแก้ตรงไหน ลองเลือกมาก่อนหนึ่งเรื่องที่อยากให้ดีขึ้นที่สุด แล้วค่อยเริ่มจากก้าวเล็ก ๆ ที่พอทำได้ครับ",
}

VENTING = (
    "ฟังแล้วรู้สึกว่าเรื่องนี้คงหนักใจอยู่ไม่น้อยเลยนะ ถ้าอยากระบายเพิ่มเติมก็ค่อย ๆ เล่าได้ครับ",
    "เรื่องนี้อาจมีหลายอย่างปนกันอยู่ เล่าเฉพาะส่วนที่อยากเล่าก่อนได้ครับ",
    "ผมรับฟังอยู่นะครับ ถ้าอยากเล่าต่อก็ค่อย ๆ เล่าได้ตามสบาย ยังไม่ต้องรีบหาทางแก้ทั้งหมดตอนนี้ก็ได้ครับ",
)

EMOTIONAL_SUPPORT = (
    "เรื่องนี้คงกระทบความรู้สึกคุณอยู่ไม่น้อยนะ ไม่ต้องรีบหาคำตอบตอนนี้ก็ได้ครับ",
    "ขอบคุณที่บอกความรู้สึกนี้ คุณเลือกได้ว่าจะเล่าต่อหรือพักการคุยไว้ก่อน",
    "ถ้าตอนนี้ยังรู้สึกหนักอยู่ ก็ไม่เป็นไรนะครับ ค่อย ๆ ให้เวลากับตัวเองก่อนก็ได้",
)

SMALLTALK = {
    "greeting": "สวัสดีครับ วันนี้อยากคุยเรื่องอะไรดี",
    "thanks": "ยินดีครับ",
    "open_chat": "คุยกันได้ครับ มีเรื่องไหนที่อยากเริ่มไหม",
}

FALLBACK = (
    "ถ้ายังไม่รู้ว่าจะเริ่มจากตรงไหน เล่าในส่วนที่อยู่ในใจก่อนก็ได้ครับ",
    "ค่อย ๆ เล่าเพิ่มได้ครับ ว่าตอนนี้มีเรื่องไหนที่อยากพูดถึงมากที่สุด",
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
        # A direct style keeps only the practical step; other styles may acknowledge emotion.
        fragments = [get_direct_advice(problem)]
        acknowledgement = (
            get_emotion_ack(emotion, choose=choose)
            if emotion != "neutral" and STYLE_RULES.get(style) != "action_first" else None
        )
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
