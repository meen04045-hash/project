from ai_model import predict_all

texts = [
    "ช่วงนี้รู้สึกเศร้ามาก",
    "ผมเครียดเรื่องเรียนมาก",
    "ไม่อยากคุยกับใครเลย",
    "ช่วงนี้เหนื่อยมาก ไม่รู้จะทำยังไง",
    "สวัสดีครับ"
]

for text in texts:
    result = predict_all(text)

    print("\n" + "=" * 60)
    print("ข้อความ:", text)
    print("=" * 60)

    print(f"Risk       : {result['risk']} ({result['risk_conf']:.3f})")
    print(f"Emotion    : {result['emotion']} ({result['emotion_conf']:.3f})")
    print(f"Problem    : {result['problem']} ({result['problem_conf']:.3f})")
    print(f"Support    : {result['support_need']} ({result['support_conf']:.3f})")
    print(f"Intent     : {result['intent']} ({result['intent_conf']:.3f})")
    print(f"Style      : {result['conversation_style']} ({result['style_conf']:.3f})")