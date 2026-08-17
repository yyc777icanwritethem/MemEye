from replay_simplemem_frozen_answers import build_messages, extract_answer, format_frozen_context


def test_format_frozen_context_preserves_official_fields():
    context = format_frozen_context(
        [
            {
                "summary": "summary",
                "details": {"full_text": "full dialogue"},
                "timestamp": 0,
                "tags": ["round_id:R1", "locomo_internal"],
                "metadata": {"persons": ["Hannah"], "location": "home"},
            }
        ]
    )
    assert "[1 memories found]" in context
    assert "Content: summary" in context
    assert "Full text: full dialogue" in context
    assert "Tags: round_id:R1" in context
    assert "locomo_internal" not in context
    assert "Persons: Hannah" in context
    assert "Location: home" in context


def test_build_messages_uses_frozen_items_without_retrieval():
    messages, multimodal = build_messages(
        {
            "question": "original",
            "recall_query": "question with frozen caption",
            "question_images": [],
            "retrieval_result": {
                "items": [
                    {
                        "id": "m1",
                        "summary": "relevant memory",
                        "timestamp": 0,
                    }
                ]
            },
        }
    )
    assert not multimodal
    assert messages[0]["content"].startswith("You are a professional Q&A assistant")
    assert "relevant memory" in messages[1]["content"]
    assert "Question: question with frozen caption" in messages[1]["content"]


def test_extract_answer_matches_official_fallbacks():
    assert extract_answer('{"reasoning":"x","answer":"sage green"}') == "sage green"
    assert extract_answer('```json\n{"answer":"two"}\n```') == "two"
