from unittest.mock import MagicMock
from src.services.judge import JudgeService
from src.services.responses import LLMResponse
from src.config.models import ModelCapability


judge_model = ModelCapability(
    id="llama3-70b-8192",
    name="Judge Model",
    description="judge",
    capabilities=[],
    limitations="none",
    best_for="judging",
)


class TestJudgeService:
    def test_judge_majority_returns_decision(self):
        cm = MagicMock()
        cm.call_with_fallback.return_value = LLMResponse(
            ok=True, text="... A"
        )
        svc = JudgeService(cm)
        result = svc.judge_majority("test prompt", "output A", "output B", judge_model)
        assert result["ok"] is True
        assert result["decision"] == "A"

    def test_judge_majority_tie(self):
        cm = MagicMock()
        cm.call_with_fallback.return_value = LLMResponse(
            ok=True, text="... TIE"
        )
        svc = JudgeService(cm)
        result = svc.judge_majority("test prompt", "out a", "out b", judge_model)
        assert result["ok"] is True
        assert result["decision"] == "TIE"

    def test_judge_majority_handles_api_failure(self):
        cm = MagicMock()
        cm.call_with_fallback.return_value = LLMResponse(ok=False, error="API down")
        svc = JudgeService(cm)
        result = svc.judge_majority("test prompt", "out a", "out b", judge_model)
        assert result["ok"] is False

    def test_judge_majority_multi_round(self):
        cm = MagicMock()
        cm.call_with_fallback.return_value = LLMResponse(
            ok=True, text="... B"
        )
        svc = JudgeService(cm)
        result = svc.judge_majority(
            "test prompt", "out a", "out b", judge_model, repeats=3
        )
        assert result["ok"] is True
        assert result["decision"] == "B"
        assert result["total_attempts"] == 3
        assert result["counts"]["A"] == 0
        assert result["counts"]["B"] == 3
