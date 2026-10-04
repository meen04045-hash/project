"""Mock-only checks for the Phase 1 ordinary response integration."""

import ast
import copy
import random
import re
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import responses


ROOT = Path(__file__).resolve().parent
APP_TREE = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))


def source_literal(name):
    for node in APP_TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing source constant: {name}")


def load_function(name, namespace):
    for node in APP_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            isolated = copy.deepcopy(node)
            isolated.decorator_list = []
            exec(compile(ast.Module(body=[isolated], type_ignores=[]), "app.py", "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"Missing source function: {name}")


def prediction(**overrides):
    values = {
        "risk": "normal", "risk_conf": 0.99,
        "emotion": "neutral", "emotion_conf": 0.1,
        "problem": "none", "problem_conf": 0.1,
        "support_need": "listener", "support_conf": 0.1,
        "intent": "smalltalk", "intent_conf": 0.1,
        "conversation_style": "friendly", "style_conf": 0.1,
    }
    values.update(overrides)
    return values


class FakeApi:
    def __init__(self, fail=False):
        self.replies = []
        self.fail = fail

    def reply_message(self, token, message):
        if self.fail:
            raise RuntimeError("mock LINE failure")
        self.replies.append((token, message))


def app_harness(text, result=None, pending=None, fail_reply=False):
    api = FakeApi(fail_reply)
    state = {"pending": pending, "writes": []}
    logs = []
    routes = []
    actual_result = result or prediction()

    def compose(*args, **kwargs):
        routes.append(kwargs["route"])
        return responses.compose_response(*args, choose=lambda items: items[0], **kwargs)

    def set_pending(_user, action):
        state["pending"] = action
        state["writes"].append(action)

    class Collection:
        def add(self, payload):
            logs.append(payload)

    namespace = {
        "random": SimpleNamespace(choice=lambda items: items[0]),
        "re": re,
        "db": SimpleNamespace(collection=lambda name: Collection()),
        "line_bot_api": api,
        "predict_all": lambda value: actual_result,
        "get_pending_action": lambda user: state["pending"],
        "set_pending_action": set_pending,
        "PENDING_ADVICE": source_literal("PENDING_ADVICE"),
        "AFFIRMATIVE_REPLIES": source_literal("AFFIRMATIVE_REPLIES"),
        "NEGATIVE_REPLIES": source_literal("NEGATIVE_REPLIES"),
        "SELF_CARE_TIPS": source_literal("SELF_CARE_TIPS"),
        "compose_response": compose,
        "select_phase1_route": responses.select_phase1_route,
        "TextSendMessage": lambda **kw: SimpleNamespace(**kw),
        "URIAction": lambda **kw: SimpleNamespace(**kw),
        "ButtonsTemplate": lambda **kw: SimpleNamespace(**kw),
        "TemplateSendMessage": lambda **kw: SimpleNamespace(**kw),
    }
    for name in (
        "reply_normal", "emotion_reply", "problem_reply", "support_reply",
        "style_reply", "detect_explicit_safety", "handle_message",
    ):
        load_function(name, namespace)
    event = SimpleNamespace(
        source=SimpleNamespace(user_id="mock-user"),
        message=SimpleNamespace(text=text),
        reply_token="mock-reply-token",
    )
    return namespace["handle_message"], event, api, state, logs, routes


class Phase1Tests(unittest.TestCase):
    def test_approved_copy_is_exact_source_for_new_response_data(self):
        approved = (ROOT / "evaluation/response_library_phase1_approved_copy.txt").read_text(encoding="utf-8")
        section = None
        approved_strings = []
        numbered_sections = {"EMOTION", "PROBLEM", "SUPPORT", "VENTING", "EMOTIONAL_SUPPORT", "FALLBACK"}
        for line in approved.splitlines():
            heading = line.split(" —", 1)[0]
            if heading in numbered_sections | {"DIRECT_ADVICE", "SMALLTALK", "STYLE_RULES"}:
                section = heading
                continue
            if section in numbered_sections and re.match(r"^  [123]\. ", line):
                approved_strings.append(line[5:])
            elif section == "DIRECT_ADVICE" and re.match(r"^(academic|work|financial|family|relationship|social|self_esteem|health|none): ", line):
                approved_strings.append(line.split(": ", 1)[1])
            elif section == "SMALLTALK" and re.match(r"^  (greeting|thanks|open_chat): ", line):
                approved_strings.append(line.split(": ", 1)[1])
        groups = (
            *responses.EMOTION_RESPONSES.values(),
            *responses.PROBLEM_CONTEXTS.values(),
            *responses.SUPPORT_RESPONSES.values(),
            responses.VENTING, responses.EMOTIONAL_SUPPORT, responses.FALLBACK,
        )
        strings = [item for group in groups for item in group]
        strings += list(responses.DIRECT_ADVICE_BY_PROBLEM.values())
        strings += list(responses.SMALLTALK.values())
        self.assertEqual(len(strings), 78)
        self.assertEqual(Counter(strings), Counter(approved_strings))
        self.assertEqual(responses.PROBLEM_CONTEXTS["none"], ())

    def test_response_module_uses_standard_library_only(self):
        tree = ast.parse((ROOT / "responses.py").read_text(encoding="utf-8"))
        imports = {
            name.name.split(".")[0]
            for node in tree.body if isinstance(node, ast.Import)
            for name in node.names
        } | {
            node.module.split(".")[0]
            for node in tree.body if isinstance(node, ast.ImportFrom)
        }
        self.assertEqual(imports, {"dataclasses", "random", "typing"})

    def test_phase1_matrix_routes_in_mocked_handle_message(self):
        cases = [
            ("RL-01", "ช่วงนี้การบ้านหลายวิชา ช่วยแนะนำหน่อยว่าจะเริ่มยังไง", dict(problem="academic", problem_conf=.99), "direct_advice"),
            ("RL-02", "งานเข้ามาพร้อมกันหลายชิ้น ควรทำยังไงก่อน", dict(problem="work", problem_conf=.99), "direct_advice"),
            ("RL-03", "รายจ่ายเดือนนี้เยอะ ช่วยแนะนำวิธีเริ่มจัดการหน่อย", dict(problem="financial", problem_conf=.99), "direct_advice"),
            ("RL-04", "ทะเลาะกับคนที่บ้าน แค่อยากให้ฟังหน่อย", dict(problem="family", problem_conf=.99, support_need="listener", support_conf=.99), "listener"),
            ("RL-05", "รู้สึกว่าตัวเองไม่เก่งเลย อยากได้กำลังใจ", dict(problem="self_esteem", problem_conf=.99, support_need="encouragement", support_conf=.99), "encouragement"),
            ("RL-06", "พรุ่งนี้ต้องพูดต่อหน้าคนเยอะ ๆ กลัวมาก ช่วยให้ใจเย็นหน่อย", dict(emotion="fear", emotion_conf=.99, support_need="calming", support_conf=.99), "calming"),
            ("RL-07", "เสียใจเรื่องเพื่อนมาก วันนี้อยากระบายเฉย ๆ", dict(emotion="sad", emotion_conf=.99, intent="venting", intent_conf=.99), "venting"),
            ("RL-08", "วันนี้รู้สึกโดดเดี่ยว ขอคำพูดให้มีกำลังใจหน่อย", dict(intent="emotional_support", intent_conf=.99, support_need="encouragement", support_conf=.99), "emotional_support"),
            ("RL-11", "สวัสดี วันนี้เป็นไงบ้าง", dict(intent="smalltalk", intent_conf=.99), "smalltalk"),
            ("RL-19", "ช่วยแนะนำหน่อย", {}, "direct_advice"),
            ("RL-20", "วันนี้ฝนตกแล้วฉันก็มีหลายเรื่อง", {}, "normal"),
            ("RL-21", "เรื่องที่บ้านกับงานปนกันไปหมด ไม่รู้จะเล่าอะไร", dict(intent="venting", intent_conf=.99), "venting"),
            ("RL-22", "อยากให้ฟังเรื่องเงินก่อน ยังไม่ต้องแนะนำ", dict(problem="financial", problem_conf=.99, support_need="advice", support_conf=.99), "listener"),
            ("RL-23", "ขอวิธีคุยกับคนรักเรื่องที่ค้างใจ", dict(problem="relationship", problem_conf=.99, intent="ask_advice", intent_conf=.99), "direct_advice"),
            ("RL-24", "เพื่อนชวนไปงานแต่รู้สึกอึดอัด ควรเริ่มบอกเขายังไง", dict(problem="social", problem_conf=.99, intent="ask_advice", intent_conf=.99), "direct_advice"),
            ("RL-26", "ไม่มีเรื่องเฉพาะ แต่อยากได้แนวทางเริ่มจัดการชีวิตหน่อย", dict(problem="none", problem_conf=.99, intent="ask_advice", intent_conf=.99), "direct_advice"),
            ("RL-27", "วันนี้เหนื่อยจากอ่านหนังสือ ช่วยแนะนำวิธีเริ่มงานที่ค้าง", dict(emotion="tired", emotion_conf=.99, problem="academic", problem_conf=.99, conversation_style="direct", style_conf=.99), "direct_advice"),
            ("RL-28", "เศร้า อยากเล่าให้ฟังหน่อย", dict(emotion="sad", emotion_conf=.99, intent="venting", intent_conf=.99, support_need="listener", support_conf=.99), "venting"),
        ]
        for case_id, text, overrides, expected_route in cases:
            with self.subTest(case_id=case_id):
                handle, event, api, _, logs, routes = app_harness(text, prediction(**overrides))
                handle(event)
                self.assertEqual(routes, [expected_route])
                self.assertEqual(len(api.replies), 1)
                self.assertTrue(api.replies[0][1].text)
                self.assertEqual(len(logs), 1)
                self.assertEqual(logs[0], {"user_id": "mock-user", "text": text, "prediction": prediction(**overrides)})
                self.assertNotIn("พร้อมรับฟัง", api.replies[0][1].text)

    def test_direct_advice_context_and_style(self):
        for phrase in responses.DIRECT_ADVICE_PHRASES[:6]:
            self.assertEqual(
                responses.select_phase1_route(phrase, eligible_intent=None, eligible_support=None),
                "direct_advice",
            )
        for problem, core in responses.DIRECT_ADVICE_BY_PROBLEM.items():
            draft = responses.compose_response("ช่วยแนะนำหน่อย", route="direct_advice", problem=problem, choose=lambda items: items[0])
            self.assertIn(core, draft.text)
            self.assertFalse(draft.offer_pending_advice)
            self.assertEqual(draft.semantic_tags, ("action",))
        self.assertEqual(
            responses.compose_response("ช่วยแนะนำหน่อย", route="direct_advice", problem="unknown").text,
            responses.DIRECT_ADVICE_BY_PROBLEM["none"],
        )
        direct = responses.compose_response(
            "ช่วยแนะนำหน่อย", route="direct_advice", problem="academic", emotion="tired",
            style="direct", choose=lambda items: items[0],
        )
        self.assertTrue(direct.text.startswith(responses.DIRECT_ADVICE_BY_PROBLEM["academic"]))
        self.assertEqual(direct.semantic_tags, ("action", "acknowledgement"))

    def test_component_limits_and_single_component_no_filler(self):
        for route in ("listener", "calming", "encouragement", "advice_offer", "normal"):
            draft = responses.compose_response(
                "เรื่องเรียน", route=route, emotion="tired", problem="academic",
                style="detailed", choose=lambda items: items[0],
            )
            self.assertEqual(len(draft.semantic_tags), len(set(draft.semantic_tags)))
            self.assertLessEqual(len(draft.semantic_tags), 4)
            self.assertLessEqual(draft.text.count("?"), 1)
        one = responses.compose_response("เรื่องเรียน", route="normal", problem="academic", choose=lambda items: items[0])
        self.assertEqual(one.text, responses.PROBLEM_CONTEXTS["academic"][0])
        self.assertEqual(one.semantic_tags, ("context",))
        unknown = responses.compose_response("ข้อความไม่ชัด", route="normal", emotion="missing", problem="missing", choose=lambda items: items[0])
        self.assertEqual(unknown.route, "fallback")
        self.assertEqual(unknown.text, responses.FALLBACK[0])
        baseline = responses.compose_response("เรื่องเรียน", route="normal", problem="academic", choose=lambda items: items[0])
        for style in ("casual", "friendly", "gentle", "motivational", "serious"):
            self.assertEqual(
                responses.compose_response("เรื่องเรียน", route="normal", problem="academic", style=style, choose=lambda items: items[0]).text,
                baseline.text,
            )

    def test_pending_and_vocabulary_regression(self):
        self.assertNotIn("ไม่ครับ", source_literal("NEGATIVE_REPLIES"))
        self.assertIn("ไม่เอา", source_literal("NEGATIVE_REPLIES"))
        yes, event, api, state, _, routes = app_harness("ได้ครับ", pending=source_literal("PENDING_ADVICE"))
        yes(event)
        self.assertEqual(api.replies[0][1].text, source_literal("SELF_CARE_TIPS"))
        self.assertEqual(state["pending"], None)
        self.assertEqual(routes, [])
        no, event, api, state, _, routes = app_harness("ไม่เอา", pending=source_literal("PENDING_ADVICE"))
        no(event)
        self.assertEqual(api.replies[0][1].text, "ผมอยู่ตรงนี้นะ ถ้าอยากเล่าอะไร 😊")
        self.assertEqual(state["pending"], None)
        self.assertEqual(routes, [])
        deferred, event, api, state, _, routes = app_harness("ไม่ครับ", pending=source_literal("PENDING_ADVICE"))
        deferred(event)
        self.assertEqual(routes, [])
        self.assertEqual(api.replies[0][1].text, "ผมอยู่ตรงนี้นะ ถ้าอยากเล่าอะไร 😊")
        self.assertEqual(state["pending"], None)
        direct, event, api, state, _, routes = app_harness("ช่วยแนะนำหน่อย", pending=source_literal("PENDING_ADVICE"))
        direct(event)
        self.assertEqual(routes, ["direct_advice"])
        self.assertIn(responses.DIRECT_ADVICE_BY_PROBLEM["none"], api.replies[0][1].text)
        self.assertEqual(state["pending"], None)
        offer, event, api, state, _, routes = app_harness("เรื่องงานเยอะ", prediction(support_need="advice", support_conf=.99))
        offer(event)
        self.assertEqual(routes, ["advice_offer"])
        self.assertEqual(state["pending"], source_literal("PENDING_ADVICE"))
        failed, event, _, state, _, _ = app_harness("เรื่องงานเยอะ", prediction(support_need="advice", support_conf=.99), fail_reply=True)
        with self.assertRaisesRegex(RuntimeError, "mock LINE failure"):
            failed(event)
        self.assertIsNone(state["pending"])

    def test_excluded_routes_preserved(self):
        for text in ("ประเมิน", "คำแนะนำการดูแลตนเอง"):
            with self.subTest(text=text):
                handle, event, api, state, logs, routes = app_harness(text, pending=source_literal("PENDING_ADVICE"))
                handle(event)
                self.assertEqual(routes, [])
                self.assertEqual(logs, [])
                self.assertEqual(state["pending"], None)
                if text == "ประเมิน":
                    self.assertEqual(api.replies[0][1].template.title, "แบบประเมินสุขภาพจิต")
                else:
                    self.assertEqual(api.replies[0][1].text, source_literal("SELF_CARE_TIPS"))
        safety_cases = [
            ("ตอนนี้อยากทำร้ายตัวเอง", prediction()),
            ("วันนี้ไม่ไหวแล้ว ขอให้ช่วยเรื่องความปลอดภัย", prediction(intent="crisis_help", intent_conf=.99)),
            ("ช่วยแนะนำเรื่องเรียนหน่อย แต่ตอนนี้อยากตาย", prediction(problem="academic", problem_conf=.99)),
            ("ความคิดต่างๆ", prediction(risk="risk_passive", risk_conf=.99)),
            ("ไม่ได้อยากตาย แต่ไม่รู้จะอยู่ไปทำไม", prediction()),
        ]
        for text, result in safety_cases:
            with self.subTest(text=text):
                handle, event, api, _, _, routes = app_harness(text, result)
                handle(event)
                self.assertEqual(routes, [])
                self.assertIn("1323", api.replies[0][1].text)
                self.assertNotIn(responses.FALLBACK[0], api.replies[0][1].text)
        safety, event, api, state, _, routes = app_harness(
            "ตอนนี้อยากทำร้ายตัวเอง", pending=source_literal("PENDING_ADVICE")
        )
        safety(event)
        self.assertEqual(routes, [])
        self.assertEqual(state["pending"], None)
        self.assertIn("1323", api.replies[0][1].text)
        clear, event, api, _, _, routes = app_harness("ตอนนี้ไม่ได้อยากตายแล้ว แต่อยากคุยเรื่องอื่น")
        clear(event)
        # The existing matcher excludes only a complete denial; a continuation stays conservative.
        self.assertEqual(routes, [])
        self.assertIn("1323", api.replies[0][1].text)
        self.assertFalse(load_function("detect_explicit_safety", {"re": re})("ตอนนี้ไม่ได้อยากตายแล้ว"))

    def test_information_and_crisis_support_preserved(self):
        info = prediction(support_need="information", support_conf=.99)
        handle, event, api, _, _, routes = app_harness("อยากรู้เรื่องนี้", info)
        handle(event)
        self.assertEqual(routes, [])
        self.assertIn("ถ้าอยากได้ข้อมูลความรู้เพิ่มเติมเกี่ยวกับสุขภาพจิต บอกผมได้เลยนะ", api.replies[0][1].text)
        info_intent = prediction(intent="information", intent_conf=.99)
        handle, event, api, _, _, routes = app_harness("ถามข้อมูล", info_intent)
        handle(event)
        self.assertEqual(routes, [])
        self.assertEqual(api.replies[0][1].text, "ผมอยู่ตรงนี้นะ ถ้าอยากเล่าอะไร 😊")
        crisis = prediction(support_need="crisis_support", support_conf=.99)
        handle, event, api, _, _, routes = app_harness("รู้สึกหนัก", crisis)
        handle(event)
        self.assertEqual(routes, [])
        self.assertIn("ถ้าคุณกำลังรู้สึกหนักใจมาก", api.replies[0][1].text)


if __name__ == "__main__":
    unittest.main()
