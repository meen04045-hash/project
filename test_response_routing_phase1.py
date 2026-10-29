"""Mock-only regression checks for ordinary response routing.

The model predictions below reproduce the local observations in the routing
audit. No LINE or Firestore client is imported or called.
"""

import unittest

import responses
from test_response_library_phase1 import app_harness, prediction, source_literal


class ResponseRoutingPhase1Tests(unittest.TestCase):
    def reply_for(self, text, overrides=None, pending=None):
        handle, event, api, state, logs, routes = app_harness(
            text, prediction(**(overrides or {})), pending=pending
        )
        handle(event)
        message = api.replies[0][1]
        return getattr(message, "text", message), routes, state, logs

    def test_nine_audit_cases_without_topic_memory(self):
        cases = [
            ("คุณชื่ออะไร", {"support_need": "information", "support_conf": .737},
             "bot_identity", ("แชทบอท", "ไม่ใช่มนุษย์")),
            ("ฉันไม่สบายใจกับเพื่อน ต้องทำยังไง",
             {"problem": "social", "problem_conf": .485, "support_need": "advice", "support_conf": .344},
             "direct_advice", ("เพื่อน", "ประเด็น")),
            ("ฉันทะเลาะกับเพื่อน",
             {"problem": "relationship", "problem_conf": .398, "intent": "smalltalk", "intent_conf": .701},
             "friend_conflict", ("เพื่อน", "รับฟัง")),
            ("ต้องทำยังไง",
             {"support_need": "advice", "support_conf": .607, "intent": "ask_advice", "intent_conf": .425},
             "direct_advice", ("หนึ่งเรื่อง", "ก้าวเล็ก")),
            ("ทำไมโง่จัง", {"conversation_style": "serious", "style_conf": .564},
             "feedback_frustrated", ("ขอโทษ", "ตรง")),
            ("ฉันบอกไปแล้ว",
             {"risk": "risk_suicide_plan", "risk_conf": .313, "intent": "crisis_help", "intent_conf": .598},
             "feedback_already_told", ("ถามซ้ำ", "ไม่ได้")),
            ("ฉันเกลียดเธอ", {"risk": "stress", "risk_conf": .717},
             "feedback_dislike", ("รับทราบ", "ไม่ชวนให้เล่าซ้ำ")),
            ("ฉันเบื่อทั้งวัน", {"emotion": "tired", "emotion_conf": .463, "intent": "smalltalk", "intent_conf": .657},
             "boredom", ("เบื่อ", "สิ่งเล็ก")),
            ("ฉันอยากให้เธอแนะนำการคิดบวก",
             {"problem": "relationship", "problem_conf": .237, "intent": "ask_advice", "intent_conf": .680},
             "positive_thinking_advice", ("ไม่จำเป็นต้องฝืน", "ก้าวเล็ก")),
        ]
        pending = None
        replies = []
        for index, (text, overrides, expected_route, expected_parts) in enumerate(cases, 1):
            with self.subTest(case=index):
                reply, routes, state, logs = self.reply_for(text, overrides, pending)
                self.assertEqual(routes, [expected_route])
                for part in expected_parts:
                    self.assertIn(part, reply)
                self.assertEqual(len(logs), 1)
                self.assertIsNone(state["pending"])
                pending = state["pending"]
                replies.append(reply)
        self.assertEqual(len(set(replies)), len(replies))

    def test_context_free_followup_does_not_invent_friend_topic(self):
        text = "ต้องทำยังไง"
        reply, routes, state, _ = self.reply_for(
            text, {"support_need": "advice", "support_conf": .607},
            pending=source_literal("PENDING_ADVICE"),
        )
        self.assertEqual(routes, ["direct_advice"])
        self.assertEqual(reply, responses.DIRECT_ADVICE_BY_PROBLEM["none"])
        self.assertNotIn("เพื่อน", reply)
        self.assertIsNone(state["pending"])

    def test_friend_context_does_not_assume_whose_feeling_it_is(self):
        self.assertFalse(responses.is_friend_conflict("เพื่อนไม่สบายใจ"))
        self.assertTrue(responses.is_friend_conflict("ฉันทะเลาะกับเพื่อน"))

    def test_friend_advice_request_low_and_high_intent(self):
        text = "ฉันอยากให้แนะนำเรื่องเพื่อน"
        for intent_conf in (.1, .99):
            with self.subTest(intent_conf=intent_conf):
                self.assertEqual(responses.select_phase1_route(
                    text,
                    eligible_intent="ask_advice" if intent_conf >= .55 else None,
                    eligible_support=None,
                ), "direct_advice")
                reply, routes, _, _ = self.reply_for(
                    text, {"intent": "ask_advice", "intent_conf": intent_conf}
                )
                self.assertEqual(routes, ["direct_advice"])
                self.assertEqual(reply, responses.FRIEND_ADVICE_CLARIFICATION)
                self.assertIn("เพื่อน", reply)
                self.assertNotIn(responses.FRIEND_CONFLICT_ADVICE, reply)

    def test_friend_advice_refusal_and_informational_context(self):
        for text in (
            "ฉันไม่อยากให้แนะนำเรื่องเพื่อน",
            "ยังไม่ต้องแนะนำเรื่องเพื่อน",
            "ฉันไม่ได้อยากให้แนะนำเรื่องเพื่อน",
        ):
            with self.subTest(refusal=text):
                self.assertEqual(responses.select_phase1_route(
                    text, eligible_intent="ask_advice", eligible_support="advice"
                ), "listener")
                reply, routes, _, _ = self.reply_for(
                    text, {"intent": "ask_advice", "intent_conf": .99}
                )
                self.assertEqual(routes, ["listener"])
                self.assertNotIn(responses.FRIEND_ADVICE_CLARIFICATION, reply)
        for text in (
            "อยากรู้เรื่องเพื่อน",
            "คำว่า อยากให้แนะนำเรื่องเพื่อน หมายถึงอะไร",
        ):
            with self.subTest(informational=text):
                self.assertFalse(responses.is_friend_advice_request(text))
                self.assertEqual(responses.select_phase1_route(
                    text, eligible_intent=None, eligible_support=None
                ), "normal")
                reply, routes, _, _ = self.reply_for(text)
                self.assertEqual(routes, ["normal"])
                self.assertNotIn(responses.FRIEND_ADVICE_CLARIFICATION, reply)
                self.assertNotIn(responses.FRIEND_CONFLICT_ACK, reply)

    def test_friend_advice_negated_fact_and_confirmed_conflict(self):
        reply, routes, _, _ = self.reply_for("ฉันไม่ได้ทะเลาะกับเพื่อน")
        self.assertEqual(routes, ["normal"])
        self.assertNotIn(responses.FRIEND_CONFLICT_ACK, reply)
        text = "ฉันไม่ได้ทะเลาะกับเพื่อน ขอคำแนะนำเรื่องเพื่อน"
        reply, routes, _, _ = self.reply_for(text)
        self.assertEqual(routes, ["direct_advice"])
        self.assertEqual(reply, responses.FRIEND_ADVICE_CLARIFICATION)
        self.assertNotIn(responses.FRIEND_CONFLICT_ADVICE, reply)
        text = "ฉันทะเลาะกับเพื่อน อยากให้แนะนำเรื่องเพื่อน"
        reply, routes, _, _ = self.reply_for(text)
        self.assertEqual(routes, ["direct_advice"])
        self.assertIn(responses.FRIEND_CONFLICT_ADVICE, reply)
        self.assertNotIn(responses.FRIEND_ADVICE_CLARIFICATION, reply)

    def test_friend_advice_request_safety_priority(self):
        cases = (
            ("ฉันอยากให้แนะนำเรื่องเพื่อน แต่อยากตาย", {}),
            ("ฉันอยากให้แนะนำเรื่องเพื่อน", {"risk": "risk_immediate_danger", "risk_conf": .99}),
            ("ฉันอยากให้แนะนำเรื่องเพื่อน", {"intent": "crisis_help", "intent_conf": .99}),
        )
        for text, overrides in cases:
            with self.subTest(text=text, overrides=overrides):
                reply, routes, state, _ = self.reply_for(
                    text, overrides, pending=source_literal("PENDING_ADVICE")
                )
                self.assertTrue(reply.startswith("🚨"))
                self.assertEqual(routes, [])
                self.assertIsNone(state["pending"])
                self.assertNotIn(responses.FRIEND_ADVICE_CLARIFICATION, reply)

    def test_refusal_routes_do_not_give_positive_thinking_advice(self):
        for text in (
            "ฉันไม่อยากให้แนะนำการคิดบวก",
            "ไม่อยากได้คำแนะนำการคิดบวก",
            "ยังไม่ต้องแนะนำ",
            "ไม่ต้องแนะนำการคิดบวก",
            "ฉันไม่สบายใจ แต่ไม่อยากให้แนะนำการคิดบวก",
            "ฉันไม่อยากให้แนะนำการคิดบวก แต่ตอนนี้อยากระบาย",
        ):
            with self.subTest(text=text):
                self.assertEqual(responses.select_phase1_route(
                    text, eligible_intent="ask_advice", eligible_support="advice"
                ), "listener")
                reply, routes, _, _ = self.reply_for(
                    text, {"intent": "ask_advice", "intent_conf": .99}
                )
                self.assertEqual(routes, ["listener"])
                self.assertNotIn(responses.POSITIVE_THINKING_ADVICE, reply)

    def test_affirmative_advice_with_unrelated_negation(self):
        for text in (
            "ฉันอยากให้แนะนำการคิดบวก",
            "ฉันไม่สบายใจ แต่อยากให้แนะนำการคิดบวก",
        ):
            with self.subTest(text=text):
                self.assertEqual(responses.select_phase1_route(
                    text, eligible_intent=None, eligible_support=None
                ), "positive_thinking_advice")
                reply, routes, _, _ = self.reply_for(text)
                self.assertEqual(routes, ["positive_thinking_advice"])
                self.assertIn(responses.POSITIVE_THINKING_ADVICE, reply)

    def test_friend_conflict_negation_information_and_affirmation(self):
        cases = (
            ("ฉันไม่ได้ทะเลาะกับเพื่อน", False),
            ("ฉันไม่ได้ไม่สบายใจกับเพื่อน", False),
            ("อยากรู้ว่าทะเลาะกับเพื่อนหมายถึงอะไร", False),
            ("คำว่า ทะเลาะกับเพื่อน หมายถึงอะไร", False),
            ("เขาทะเลาะกับเพื่อน", False),
            ("เพื่อนฉันทะเลาะกับเพื่อน", False),
            ("ฉันทะเลาะกับเพื่อน", True),
            ("ทะเลาะกับเพื่อนสนิท", True),
            ("ฉันไม่สบายใจกับเพื่อน", True),
            ("ฉันไม่ได้ทะเลาะกับเพื่อน แต่ตอนนี้ทะเลาะกับเพื่อนสนิท", True),
        )
        for text, has_conflict in cases:
            with self.subTest(text=text):
                self.assertEqual(responses.is_friend_conflict(text), has_conflict)
                expected_route = "friend_conflict" if has_conflict else "normal"
                self.assertEqual(responses.select_phase1_route(
                    text, eligible_intent=None, eligible_support=None
                ), expected_route)
                reply, routes, _, _ = self.reply_for(text)
                self.assertEqual(routes, [expected_route])
                self.assertEqual(responses.FRIEND_CONFLICT_ACK in reply, has_conflict)

    def test_direct_advice_uses_only_confirmed_current_friend_context(self):
        cases = (
            ("ฉันไม่ได้ทะเลาะกับเพื่อน ต้องทำยังไง", False),
            ("ฉันไม่ได้ไม่สบายใจกับเพื่อน ต้องทำยังไง", False),
            ("อยากรู้ว่าทะเลาะกับเพื่อนหมายถึงอะไร ต้องทำยังไง", False),
            ("ฉันไม่สบายใจกับเพื่อน ต้องทำยังไง", True),
            ("ทะเลาะกับเพื่อนสนิท ต้องทำยังไง", True),
            ("ฉันไม่ได้ทะเลาะกับเพื่อน แต่ไม่สบายใจกับเพื่อน ต้องทำยังไง", True),
        )
        for text, has_conflict in cases:
            with self.subTest(text=text):
                self.assertEqual(responses.is_friend_conflict(text), has_conflict)
                self.assertEqual(responses.select_phase1_route(
                    text, eligible_intent=None, eligible_support=None
                ), "direct_advice")
                reply, routes, _, _ = self.reply_for(text)
                self.assertEqual(routes, ["direct_advice"])
                self.assertEqual(responses.FRIEND_CONFLICT_ADVICE in reply, has_conflict)
                if not has_conflict:
                    self.assertEqual(reply, responses.DIRECT_ADVICE_BY_PROBLEM["none"])

    def test_safety_overrides_refusal_with_risk_signal(self):
        text = "ฉันไม่อยากให้แนะนำการคิดบวก"
        cases = (
            (text + " แต่อยากตาย", {}),
            (text, {"risk": "risk_immediate_danger", "risk_conf": .99}),
            (text, {"intent": "crisis_help", "intent_conf": .99}),
        )
        for message, overrides in cases:
            with self.subTest(text=message, overrides=overrides):
                reply, routes, state, _ = self.reply_for(
                    message, overrides, pending=source_literal("PENDING_ADVICE")
                )
                self.assertTrue(reply.startswith("🚨"))
                self.assertEqual(routes, [])
                self.assertIsNone(state["pending"])

    def test_feedback_clears_pending_without_repeating_invitation(self):
        for text, expected in (
            ("ทำไมโง่จัง", "feedback_frustrated"),
            ("ฉันบอกไปแล้ว", "feedback_already_told"),
            ("ฉันเกลียดเธอ", "feedback_dislike"),
        ):
            with self.subTest(text=text):
                reply, routes, state, _ = self.reply_for(
                    text, pending=source_literal("PENDING_ADVICE")
                )
                self.assertEqual(routes, [expected])
                self.assertIsNone(state["pending"])
                self.assertNotIn(reply, responses.FALLBACK)

    def test_listener_opt_out_overrides_advice_request(self):
        text = "ฉันทะเลาะกับเพื่อน ยังไม่ต้องแนะนำ"
        reply, routes, _, _ = self.reply_for(
            text, {"problem": "social", "problem_conf": .99, "intent": "ask_advice", "intent_conf": .99}
        )
        self.assertEqual(routes, ["listener"])
        self.assertNotIn(responses.FRIEND_CONFLICT_ADVICE, reply)
        self.assertEqual(
            responses.select_phase1_route(
                "ไม่อยากได้คำแนะนำการคิดบวก",
                eligible_intent="ask_advice",
                eligible_support="advice",
            ),
            "listener",
        )

    def test_safety_precedes_advice_and_feedback(self):
        cases = [
            ("อยากตาย ต้องทำยังไง", {}),
            ("ฉันไม่สบายใจกับเพื่อน ต้องทำยังไง",
             {"risk": "risk_immediate_danger", "risk_conf": .99}),
            ("ฉันอยากให้เธอแนะนำการคิดบวก",
             {"intent": "crisis_help", "intent_conf": .99}),
            ("ฉันบอกไปแล้ว อยากทำร้ายตัวเอง", {}),
        ]
        for text, overrides in cases:
            with self.subTest(text=text):
                reply, routes, state, _ = self.reply_for(
                    text, overrides, pending=source_literal("PENDING_ADVICE")
                )
                self.assertTrue(reply.startswith("🚨"))
                self.assertEqual(routes, [])
                self.assertIsNone(state["pending"])

    def test_clear_negation_and_mixed_safety_behavior(self):
        clear, _, _, _ = self.reply_for("ไม่ได้อยากตาย")
        mixed, _, _, _ = self.reply_for("ไม่ได้อยากตาย แต่ไม่รู้จะอยู่ไปทำไม")
        self.assertFalse(clear.startswith("🚨"))
        self.assertTrue(mixed.startswith("🚨"))

    def test_existing_exact_commands_still_bypass_ordinary_routing(self):
        _, routes, state, logs = self.reply_for(
            "ประเมิน", pending=source_literal("PENDING_ADVICE")
        )
        self.assertEqual(routes, [])
        self.assertEqual(logs, [])
        self.assertIsNone(state["pending"])
        reply, routes, state, logs = self.reply_for(
            "คำแนะนำการดูแลตนเอง", pending=source_literal("PENDING_ADVICE")
        )
        self.assertEqual(reply, source_literal("SELF_CARE_TIPS"))
        self.assertEqual(routes, [])
        self.assertEqual(logs, [])
        self.assertIsNone(state["pending"])


class ConversationClosingTests(unittest.TestCase):
    reply_for = ResponseRoutingPhase1Tests.reply_for

    def test_clear_thanks_closing_independent_of_ordinary_ml_labels(self):
        texts = (
            "ขอบคุณครับ เดี๋ยวจะลองทำทีละอย่างดู",
            "ขอบคุณมากนะ",
            "โอเคครับ ขอบคุณสำหรับคำแนะนำ",
        )
        for text in texts:
            for overrides in ({}, {"intent": "ask_advice", "intent_conf": .99},
                              {"intent": "information", "intent_conf": .99},
                              {"support_need": "advice", "support_conf": .99}):
                with self.subTest(text=text, overrides=overrides):
                    reply, routes, state, _ = self.reply_for(text, overrides)
                    self.assertEqual(routes, ["conversation_closing"])
                    self.assertIn(reply, responses.CONVERSATION_CLOSINGS.values())
                    self.assertNotIn(reply, responses.FALLBACK)
                    self.assertIsNone(state["pending"])
                    self.assertNotIn("?", reply)
                    self.assertNotIn("invitation", responses.compose_response(text, route="conversation_closing").semantic_tags)

    def test_short_thanks(self):
        for text in ("ขอบคุณ", "ขอบคุณครับ", "ขอบคุณค่ะ!", " ขอบคุณมากนะ 🙏 "):
            with self.subTest(text=text):
                reply, routes, _, _ = self.reply_for(text)
                self.assertEqual(routes, ["conversation_closing"])
                self.assertEqual(reply, responses.CONVERSATION_CLOSINGS["thanks"])

    def test_goodbye(self):
        for text in ("ไว้คุยกันใหม่นะ", "ไปก่อนนะครับ"):
            with self.subTest(text=text):
                reply, routes, _, _ = self.reply_for(text)
                self.assertEqual(routes, ["conversation_closing"])
                self.assertEqual(reply, responses.CONVERSATION_CLOSINGS["goodbye"])

    def test_thanks_with_continuing_request_is_not_closed(self):
        for text in (
            "ขอบคุณนะ แต่ตอนนี้ยังไม่รู้จะทำยังไง ช่วยแนะนำอีกหน่อย",
            "ขอบคุณครับ แล้วควรทำยังไงต่อ",
            "ขอบคุณนะ ช่วยแนะนำหน่อย",
            "ขอบคุณมากนะ อยากระบายต่อ",
            "คำว่า ขอบคุณมากนะ หมายถึงอะไร",
            "ไปก่อนนะครับ แต่ช่วยแนะนำหน่อย",
        ):
            with self.subTest(text=text):
                self.assertIsNone(responses.conversation_closing_kind(text))
                for overrides in ({}, {"intent": "smalltalk", "intent_conf": .99}):
                    reply, routes, _, _ = self.reply_for(text, overrides)
                    self.assertNotIn("conversation_closing", routes)
                    self.assertNotIn(reply, responses.CONVERSATION_CLOSINGS.values())
                    self.assertNotEqual(reply, responses.SMALLTALK["thanks"])

    def test_safety_overrides_thanks_and_goodbye(self):
        cases = (
            ("ขอบคุณครับ แต่ตอนนี้อยากตาย", {}),
            ("ไปก่อนนะครับ อยากทำร้ายตัวเอง", {}),
            ("ขอบคุณมากนะ", {"risk": "risk_immediate_danger", "risk_conf": .99}),
            ("ขอบคุณครับ เดี๋ยวจะลองทำทีละอย่างดู", {"risk": "risk_suicide_plan", "risk_conf": .99}),
            ("ไว้คุยกันใหม่นะ", {"intent": "crisis_help", "intent_conf": .99}),
        )
        for text, overrides in cases:
            with self.subTest(text=text, overrides=overrides):
                reply, routes, state, _ = self.reply_for(text, overrides, source_literal("PENDING_ADVICE"))
                self.assertTrue(reply.startswith("🚨"))
                self.assertEqual(routes, [])
                self.assertIsNone(state["pending"])

    def test_closing_clears_pending_advice_without_new_offer(self):
        reply, routes, state, logs = self.reply_for(
            "ขอบคุณครับ เดี๋ยวจะลองทำทีละอย่างดู", pending=source_literal("PENDING_ADVICE"))
        self.assertEqual(routes, ["conversation_closing"])
        self.assertEqual(reply, responses.CONVERSATION_CLOSINGS["next_step"])
        self.assertIsNone(state["pending"])
        self.assertEqual(state["writes"], [None])
        self.assertEqual(len(logs), 1)

    def test_existing_continuing_routes(self):
        cases = (
            ("ขอบคุณนะ แค่อยากให้ฟัง", {}, "listener"),
            ("ขอบคุณนะ ช่วยแนะนำหน่อย", {}, "direct_advice"),
            ("ขอบคุณนะ อยากระบายต่อ", {}, "venting"),
            ("ขอบคุณนะ อยากรู้เรื่องนี้", {"intent": "information", "intent_conf": .99}, None),
            ("ขอบคุณมากนะ", {"support_need": "crisis_support", "support_conf": .99}, None),
        )
        for text, overrides, expected in cases:
            with self.subTest(text=text):
                _, routes, _, _ = self.reply_for(text, overrides)
                self.assertEqual(routes, [expected] if expected else [])

    def test_positive_statements_do_not_prove_risk_resolution(self):
        self.assertIsNone(responses.conversation_closing_kind("ตอนนี้ดีขึ้นแล้ว"))
        reply, routes, _, _ = self.reply_for(
            "ตอนนี้ดีขึ้นแล้ว", {"risk": "risk_immediate_danger", "risk_conf": .99})
        self.assertTrue(reply.startswith("🚨"))
        self.assertEqual(routes, [])


if __name__ == "__main__":
    unittest.main()
