from src.hallucination_detector import HallucinationDetector, parse_judge_response


def test_detects_supported_claim_with_local_fallback():
    detector = HallucinationDetector(use_llm_judge=False)
    contexts = ["LoRA reduces trainable parameters by injecting low-rank matrices into model weights."]
    answer = "LoRA reduces trainable parameters by injecting low-rank matrices."

    result = detector.detect("What problem does LoRA solve?", contexts, answer)

    assert result.final_judgement == "supported"
    assert result.hallucination_rate == 0.0
    assert result.spans[0].status == "supported"


def test_detects_unsupported_claim_with_local_fallback():
    detector = HallucinationDetector(use_llm_judge=False)
    contexts = ["RAG combines retrieval with generation to ground answers in external documents."]
    answer = "RAG was invented by Alan Turing in 1950."

    result = detector.detect("Who invented RAG?", contexts, answer)

    assert result.final_judgement == "unsupported"
    assert result.hallucination_rate == 1.0
    assert result.spans[0].status == "unsupported"


def test_parse_judge_response_handles_json_wrapped_in_text():
    raw = """
    Here is the judgement:
    {
      "claims": [
        {"text": "RAG retrieves context.", "status": "supported", "reason": "Context says retrieval is used."},
        {"text": "RAG always eliminates hallucinations.", "status": "unsupported", "reason": "No such guarantee is provided."}
      ],
      "final_judgement": "partially_supported"
    }
    """

    result = parse_judge_response(raw)

    assert result.final_judgement == "partially_supported"
    assert result.hallucination_rate == 0.5
    assert [span.status for span in result.spans] == ["supported", "unsupported"]


def test_parse_judge_response_falls_back_to_unclear():
    result = parse_judge_response("not json")

    assert result.final_judgement == "unsupported"
    assert result.hallucination_rate == 0.0
    assert result.spans[0].status == "unclear"
