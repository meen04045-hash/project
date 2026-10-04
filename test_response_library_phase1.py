"""Mock-only checks for the Phase 1 ordinary response integration."""

import ast
import copy
import random
import re
import subprocess
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
    def test_round3_legacy_copy_and_exclusions(self):
        audit = (ROOT / "evaluation/response_library_natural_tone_review.txt").read_text(encoding="utf-8")
        inventory = audit.split("2. COMPLETE SOURCE INVENTORY AND INDIVIDUAL RATINGS", 1)[1].split("3. REWRITE DRAFT", 1)[0]
        rewrites = audit.split("3. REWRITE DRAFT", 1)[1].split("4. CATEGORY SUMMARY", 1)[0]
        entries = re.findall(
            r"^(L-[ENPSY]:[^|\n]+) \| rating ([BCD]) \| app\.py::([a-z_]+) \| ([a-z_.]+)\n"
            r"CURRENT: (.*)\nISSUE: .*\nPROPOSED: (.*)\nRATIONALE: ",
            rewrites, re.M,
        )
        self.assertEqual(len(entries), 49)
        functions = {node.name: node for node in APP_TREE.body if isinstance(node, ast.FunctionDef)}
        mapping_name = {
            "emotion_reply": "emotion_map",
            "problem_reply": "mapping",
            "support_reply": "mapping",
            "style_reply": "mapping",
        }
        expected_function = {
            "L-E": "emotion_reply", "L-P": "problem_reply", "L-S": "support_reply",
            "L-Y": "style_reply", "L-N": "reply_normal",
        }
        for item_id, grade, function, label, current, proposed in entries:
            with self.subTest(item_id=item_id):
                prefix, key, index = item_id.split(":")
                self.assertIn(grade, "BCD")
                self.assertEqual(function, expected_function[prefix])
                self.assertEqual(label, "reply_normal." + key if prefix == "L-N" else key)
                literals = [
                    node.value for node in ast.walk(functions[function])
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)
                ]
                self.assertEqual(literals.count(proposed), 1)
                self.assertNotIn(current, literals)
                if prefix != "L-N":
                    assignment = next(
                        node for node in ast.walk(functions[function])
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == mapping_name[function]
                            for target in node.targets
                        )
                    )
                    self.assertEqual(ast.literal_eval(assignment.value)[key][int(index) - 1], proposed)

        a_entries = re.findall(
            r"^(L-[ENPSY]:[^|\n]+) \| A \| app\.py::([a-z_]+) \| .*? \| CURRENT: (.*)$",
            inventory, re.M,
        )
        self.assertEqual(len(a_entries), 41)
        for item_id, function, original in a_entries:
            literals = [
                node.value for node in ast.walk(functions[function])
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            ]
            self.assertIn(original, literals, item_id)

        baseline_source = subprocess.check_output(
            ["git", "show", "c72303184e88c285bfc60b27dd8669ce98780071:app.py"],
            cwd=ROOT,
        ).decode("utf-8")
        baseline = ast.parse(baseline_source)
        baseline_functions = {
            node.name: node for node in baseline.body if isinstance(node, ast.FunctionDef)
        }
        for name in ("SELF_CARE_TIPS", "AFFIRMATIVE_REPLIES", "NEGATIVE_REPLIES"):
            baseline_literal = next(
                ast.literal_eval(node.value) for node in baseline.body
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == name for target in node.targets
                )
            )
            self.assertEqual(source_literal(name), baseline_literal)
        def support_mapping(function):
            return ast.literal_eval(next(
                node.value for node in ast.walk(function)
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "mapping"
                    for target in node.targets
                )
            ))
        self.assertEqual(
            support_mapping(functions["support_reply"])["crisis_support"],
            support_mapping(baseline_functions["support_reply"])["crisis_support"],
        )
        for name in ("detect_explicit_safety", "validate_assessment_answers", "build_assessment_result"):
            self.assertEqual(
                ast.dump(functions[name], include_attributes=False),
                ast.dump(baseline_functions[name], include_attributes=False),
            )
        def safety_body(function):
            return next(
                node.body for node in ast.walk(function)
                if isinstance(node, ast.If)
                and isinstance(node.test, ast.BoolOp)
                and isinstance(node.test.op, ast.Or)
                and {part.id for part in node.test.values if isinstance(part, ast.Name)}
                == {"explicit_safety", "risk_is_high", "intent_is_crisis"}
            )
        self.assertEqual(
            ast.dump(ast.Module(body=safety_body(functions["handle_message"]), type_ignores=[]), include_attributes=False),
            ast.dump(ast.Module(body=safety_body(baseline_functions["handle_message"]), type_ignores=[]), include_attributes=False),
        )

    def test_unchanged_phase1_copy_matches_original_approved_copy(self):
        approved = (ROOT / "evaluation/response_library_phase1_approved_copy.txt").read_text(encoding="utf-8")
        section = None
        label = None
        approved_strings = {}
        numbered_sections = {"EMOTION", "PROBLEM", "FALLBACK"}
        for line in approved.splitlines():
            heading = line.split(" —", 1)[0]
            if heading in numbered_sections | {"SUPPORT", "DIRECT_ADVICE", "VENTING", "EMOTIONAL_SUPPORT", "SMALLTALK", "STYLE_RULES"}:
                section = heading
                label = None
                continue
            if section in {"EMOTION", "PROBLEM"} and re.match(r"^[a-z_]+(?: —.*)?$", line):
                label = line.split(" —", 1)[0]
            elif section in numbered_sections and re.match(r"^  [123]\. ", line):
                approved_strings[(section, label, int(line[2]))] = line[5:]
            elif section == "SMALLTALK" and re.match(r"^  (greeting|thanks|open_chat): ", line):
                key, value = line.strip().split(": ", 1)
                approved_strings[("SMALLTALK", key, 1)] = value
        changed = {
            ("EMOTION", "fear", 2), ("EMOTION", "happy", 3),
            ("EMOTION", "neutral", 2), ("EMOTION", "sad", 2),
            ("EMOTION", "tired", 2),
            ("PROBLEM", "academic", 1), ("PROBLEM", "financial", 1),
            ("PROBLEM", "family", 3), ("PROBLEM", "self_esteem", 1),
            ("PROBLEM", "health", 3),
            ("FALLBACK", None, 1), ("FALLBACK", None, 2),
        }
        self.assertEqual(len(approved_strings), 51)
        self.assertEqual(len(changed), 12)
        for (group, key, index), original in approved_strings.items():
            if (group, key, index) in changed:
                continue
            current = {
                "EMOTION": responses.EMOTION_RESPONSES,
                "PROBLEM": responses.PROBLEM_CONTEXTS,
                "FALLBACK": {None: responses.FALLBACK},
                "SMALLTALK": {key: (responses.SMALLTALK[key],)} if group == "SMALLTALK" else {},
            }[group][key][index - 1]
            self.assertEqual(current, original, (group, key, index))
        self.assertEqual(responses.PROBLEM_CONTEXTS["none"], ())

    def test_round1_approved_supportive_copy(self):
        self.assertEqual(responses.VENTING, (
            "ฟังแล้วรู้สึกว่าเรื่องนี้คงหนักใจอยู่ไม่น้อยเลยนะ ถ้าอยากระบายเพิ่มเติมก็ค่อย ๆ เล่าได้ครับ",
            "เรื่องนี้อาจมีหลายอย่างปนกันอยู่ เล่าเฉพาะส่วนที่อยากเล่าก่อนได้ครับ",
            "ผมรับฟังอยู่นะครับ ถ้าอยากเล่าต่อก็ค่อย ๆ เล่าได้ตามสบาย ยังไม่ต้องรีบหาทางแก้ทั้งหมดตอนนี้ก็ได้ครับ",
        ))
        self.assertEqual(responses.SUPPORT_RESPONSES["listener"], (
            "เล่าเท่าที่สบายใจได้เลยครับ ผมฟังอยู่",
            "ไม่ต้องเรียบเรียงให้ครบก็ได้ครับ ค่อย ๆ เล่าในส่วนที่อยากเล่าก็พอ",
            "ถ้ามีเรื่องไหนที่ยังค้างอยู่ในใจ เล่าให้ฟังได้ครับ ไม่ต้องรีบ",
        ))
        self.assertEqual(responses.SUPPORT_RESPONSES["calming"], (
            "งั้นลองค่อย ๆ หายใจเข้า–ออกช้า ๆ ดูนะครับ อาจช่วยให้รู้สึกผ่อนคลายขึ้นได้บ้าง",
            "ลองพักสักครู่ แล้วค่อย ๆ มองสิ่งรอบตัวทีละอย่างนะครับ อาจช่วยให้รู้สึกตั้งหลักได้มากขึ้น",
            "ยังไม่ต้องรีบจัดการทุกอย่างตอนนี้ก็ได้นะครับ ลองพักสักครู่ก่อน แล้วค่อยกลับมาดูทีละเรื่องก็ได้",
        ))
        self.assertEqual(responses.SUPPORT_RESPONSES["encouragement"], (
            "ค่อย ๆ ทำทีละเรื่องก็พอนะครับ ตอนนี้ยังไม่จำเป็นต้องมั่นใจกับทุกอย่างก็ได้",
            "ยังไม่ต้องรีบทำทุกอย่างให้ได้ในครั้งเดียวครับ เริ่มจากสิ่งเล็ก ๆ ที่พอไหวก่อนก็ได้",
            "ถึงตอนนี้บางอย่างอาจยังไม่เป็นอย่างที่หวัง ก็ไม่ได้แปลว่าคุณทำได้ไม่ดีนะครับ ค่อย ๆ ให้เวลากับตัวเองแล้วไปทีละขั้นก็พอ",
        ))
        self.assertEqual(responses.EMOTIONAL_SUPPORT[:2], (
            "เรื่องนี้คงกระทบความรู้สึกคุณอยู่ไม่น้อยนะ ไม่ต้องรีบหาคำตอบตอนนี้ก็ได้ครับ",
            "ขอบคุณที่บอกความรู้สึกนี้ คุณเลือกได้ว่าจะเล่าต่อหรือพักการคุยไว้ก่อน",
        ))
        self.assertEqual(responses.DIRECT_ADVICE_BY_PROBLEM, {
            "academic": "ถ้าเรื่องเรียนตอนนี้มีหลายอย่างเข้ามาพร้อมกัน ลองจดงานที่ต้องทำออกมาก่อน แล้วค่อยเลือกเริ่มจากงานที่ใกล้กำหนดที่สุดก็ได้ครับ",
            "work": "ถ้างานตอนนี้มีหลายอย่างจนไม่รู้จะเริ่มตรงไหน ลองแยกก่อนว่าอะไรต้องทำวันนี้ แล้วค่อยเริ่มจากเรื่องที่สำคัญที่สุดสักหนึ่งอย่างครับ",
            "financial": "ถ้าช่วงนี้เรื่องค่าใช้จ่ายทำให้กังวล ลองเขียนรายรับกับค่าใช้จ่ายที่จำเป็นออกมาก่อน จะได้ค่อย ๆ ดูว่าเรื่องไหนควรจัดการก่อนครับ",
            "family": "ถ้าเป็นเรื่องในครอบครัว ลองเลือกคุยทีละประเด็นก่อนก็ได้ครับ แล้วหาจังหวะที่ทั้งสองฝ่ายพร้อมคุยกันมากที่สุด",
            "relationship": "ถ้ายังไม่แน่ใจว่าจะเริ่มคุยกับอีกฝ่ายยังไง ลองเรียบเรียงสิ่งที่อยากบอกไว้ก่อนสักนิดก็ได้ครับ จะได้พูดสิ่งที่สำคัญกับคุณได้ชัดขึ้น",
            "social": "ถ้าเป็นเรื่องกับคนรอบตัว ลองคิดก่อนว่ามีเรื่องไหนที่อยากให้อีกฝ่ายเข้าใจมากที่สุด แล้วค่อยเริ่มคุยจากเรื่องนั้นครับ",
            "self_esteem": "ถ้าตอนนี้กำลังรู้สึกว่าตัวเองทำได้ไม่ดีพอ ลองแยกก่อนว่าอะไรคือสิ่งที่เกิดขึ้นจริง กับอะไรที่เป็นคำตัดสินตัวเอง แล้วค่อยดูว่าส่วนไหนที่พอแก้ได้ครับ",
            "health": "ถ้ามีเรื่องสุขภาพที่ทำให้กังวล ลองจดอาการหรือสิ่งที่สงสัยไว้ก่อนนะครับ แล้วค่อยนำไปปรึกษาผู้เชี่ยวชาญที่เหมาะสม",
            "none": "ถ้ายังไม่รู้ว่าจะเริ่มแก้ตรงไหน ลองเลือกมาก่อนหนึ่งเรื่องที่อยากให้ดีขึ้นที่สุด แล้วค่อยเริ่มจากก้าวเล็ก ๆ ที่พอทำได้ครับ",
        })

    def test_round2_approved_copy(self):
        for label, index, expected in (
            ("fear", 1, "ฟังดูแล้วเรื่องนี้ชวนให้กังวลอยู่เหมือนกันนะ"),
            ("happy", 2, "ดูเป็นเรื่องที่มีความหมายกับคุณมากนะ"),
            ("neutral", 1, "ได้ครับ ค่อย ๆ เล่าต่อได้ตามที่สบายใจนะ"),
            ("sad", 1, "เจอเรื่องแบบนี้แล้วรู้สึกเสียใจได้เหมือนกันนะ"),
            ("tired", 1, "ฟังดูว่าช่วงนี้คุณเหนื่อยอยู่มากเลยนะ"),
        ):
            self.assertEqual(responses.EMOTION_RESPONSES[label][index], expected)
        for label, index, expected in (
            ("academic", 0, "เรื่องเรียนช่วงนี้ดูจะทำให้คุณต้องรับมือกับหลายอย่างเลยนะ"),
            ("financial", 0, "เรื่องค่าใช้จ่ายช่วงนี้คงทำให้ต้องคิดหลายอย่างอยู่เหมือนกันนะ"),
            ("family", 2, "เรื่องในบ้านบางทีก็กระทบความรู้สึกเราได้มากจริง ๆ"),
            ("self_esteem", 0, "เรื่องนี้อาจทำให้คุณตั้งคำถามกับตัวเองมากขึ้น"),
            ("health", 2, "ความไม่แน่ใจเรื่องสุขภาพแบบนี้ทำให้กังวลได้เหมือนกันนะ"),
        ):
            self.assertEqual(responses.PROBLEM_CONTEXTS[label][index], expected)
        self.assertEqual(responses.SUPPORT_RESPONSES["advice"], (
            "ถ้าต้องการ เราลองคิดทางเลือกเบื้องต้นกันได้",
            "ถ้าอยากลองหาทางออก เราเริ่มจากเรื่องที่ติดอยู่ที่สุดก่อนได้ครับ",
            "ถ้าอยากให้ช่วยคิดขั้นแรก บอกได้ว่าติดตรงไหน",
        ))
        self.assertEqual(responses.FALLBACK, (
            "ถ้ายังไม่รู้ว่าจะเริ่มจากตรงไหน เล่าในส่วนที่อยู่ในใจก่อนก็ได้ครับ",
            "ค่อย ๆ เล่าเพิ่มได้ครับ ว่าตอนนี้มีเรื่องไหนที่อยากพูดถึงมากที่สุด",
            "ถ้าอยากได้คำแนะนำหรือแค่อยากระบาย บอกแบบที่สะดวกได้ครับ",
        ))
        self.assertEqual(responses.EMOTIONAL_SUPPORT, (
            "เรื่องนี้คงกระทบความรู้สึกคุณอยู่ไม่น้อยนะ ไม่ต้องรีบหาคำตอบตอนนี้ก็ได้ครับ",
            "ขอบคุณที่บอกความรู้สึกนี้ คุณเลือกได้ว่าจะเล่าต่อหรือพักการคุยไว้ก่อน",
            "ถ้าตอนนี้ยังรู้สึกหนักอยู่ ก็ไม่เป็นไรนะครับ ค่อย ๆ ให้เวลากับตัวเองก่อนก็ได้",
        ))

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
        self.assertEqual(direct.semantic_tags, ("action",))

    def test_tone_source_fix_composed_variants_and_direct_advice(self):
        for index in range(3):
            choose = lambda items, index=index: items[index]
            calming = responses.compose_response(
                "พรุ่งนี้ต้องนำเสนองาน กลัวมาก", route="calming", emotion="fear",
                problem="academic", style="gentle", choose=choose,
            )
            self.assertEqual(
                calming.text,
                responses.EMOTION_RESPONSES["fear"][index] + " "
                + responses.SUPPORT_RESPONSES["calming"][index],
            )
            self.assertLessEqual(calming.text.count("ฟังดู"), 1)
            self.assertNotIn("ฟังดูว่าตอนนี้คุณคงกังวล", calming.text)
            encouragement = responses.compose_response(
                "รู้สึกว่าตัวเองไม่เก่ง", route="encouragement", emotion="sad",
                problem="self_esteem", style="gentle", choose=choose,
            )
            self.assertEqual(
                encouragement.text,
                responses.PROBLEM_CONTEXTS["self_esteem"][index] + " "
                + responses.SUPPORT_RESPONSES["encouragement"][index],
            )
            self.assertNotIn("ต้องรับมือกับหลายอย่างพร้อมกัน", encouragement.text)
        for key, emotion in (("academic", "tired"), ("financial", "fear"), ("self_esteem", "sad")):
            direct = responses.compose_response(
                "ช่วยแนะนำหน่อย", route="direct_advice", emotion=emotion,
                problem=key, style="direct", choose=lambda items: items[0],
            )
            self.assertEqual(direct.text, responses.DIRECT_ADVICE_BY_PROBLEM[key])
            self.assertEqual(direct.semantic_tags, ("action",))
        ordinary_style = responses.compose_response(
            "ช่วยแนะนำหน่อย", route="direct_advice", emotion="tired",
            problem="academic", style="gentle", choose=lambda items: items[0],
        )
        self.assertTrue(ordinary_style.text.startswith(responses.EMOTION_RESPONSES["tired"][0]))
        self.assertIn(responses.DIRECT_ADVICE_BY_PROBLEM["academic"], ordinary_style.text)

    def test_tone_source_fix_legacy_information_and_pending_scope(self):
        information = prediction(
            emotion="fear", emotion_conf=.99, problem="academic", problem_conf=.99,
            support_need="information", support_conf=.99,
            intent="information", intent_conf=.99,
            conversation_style="gentle", style_conf=.99,
        )
        handle, event, api, _, _, routes = app_harness("อยากรู้วิธีดูแลใจเวลารู้สึกเครียด", information)
        handle(event)
        reply = api.replies[0][1].text
        self.assertEqual(routes, [])
        self.assertIn("ลองพักจากสิ่งที่ทำอยู่สักครู่", reply)
        self.assertNotIn("เรื่องเรียน", reply)
        self.assertNotIn("บอกหัวข้อ", reply)
        self.assertNotIn("ค่อย ๆ คุยไปทีละเรื่อง", reply)
        self.assertEqual(len(reply.split("\n\n")), 2)
        family = prediction(
            emotion="sad", emotion_conf=.99, problem="family", problem_conf=.99,
            support_need="listener", support_conf=.99,
            intent="venting", intent_conf=.99,
            conversation_style="gentle", style_conf=.99,
        )
        handle, event, api, state, _, routes = app_harness(
            "ทะเลาะกับคนในครอบครัว อยากเล่า", family, pending=source_literal("PENDING_ADVICE")
        )
        handle(event)
        self.assertEqual(routes, [])
        self.assertIsNone(state["pending"])
        self.assertEqual(len(api.replies[0][1].text.split("\n\n")), 2)
        self.assertIn("ครอบครัว", api.replies[0][1].text)
        self.assertNotIn("ค่อย ๆ คุยไปทีละเรื่อง", api.replies[0][1].text)
        for route in ("listener", "venting", "emotional_support", "normal"):
            draft = responses.compose_response(
                "แค่อยากเล่า", route=route, emotion=None, problem=None,
                choose=lambda items: items[0],
            )
            self.assertTrue(draft.text)
        self.assertNotIn("ไม่ครับ", source_literal("NEGATIVE_REPLIES"))

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
        self.assertEqual(api.replies[0][1].text, "ถ้าอยากเล่าต่อ ผมฟังอยู่นะครับ")
        self.assertEqual(state["pending"], None)
        self.assertEqual(routes, [])
        deferred, event, api, state, _, routes = app_harness("ไม่ครับ", pending=source_literal("PENDING_ADVICE"))
        deferred(event)
        self.assertEqual(routes, [])
        self.assertEqual(api.replies[0][1].text, "ถ้าอยากเล่าต่อ ผมฟังอยู่นะครับ")
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
        self.assertIn("ถ้าอยากรู้เรื่องการดูแลใจด้านไหน บอกหัวข้อได้ครับ", api.replies[0][1].text)
        info_intent = prediction(intent="information", intent_conf=.99)
        handle, event, api, _, _, routes = app_harness("ถามข้อมูล", info_intent)
        handle(event)
        self.assertEqual(routes, [])
        self.assertEqual(api.replies[0][1].text, "ถ้าอยากเล่าต่อ ผมฟังอยู่นะครับ")
        crisis = prediction(support_need="crisis_support", support_conf=.99)
        handle, event, api, _, _, routes = app_harness("รู้สึกหนัก", crisis)
        handle(event)
        self.assertEqual(routes, [])
        self.assertIn("ถ้าคุณกำลังรู้สึกหนักใจมาก", api.replies[0][1].text)


if __name__ == "__main__":
    unittest.main()
