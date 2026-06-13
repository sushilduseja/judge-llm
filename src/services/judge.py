from typing import Dict, Optional, List
import re
from .client_manager import ClientManager
from ..config.models import ModelCapability


class JudgeService:
    def __init__(self, client_manager: ClientManager):
        self.client_manager = client_manager

    def _create_judge_prompt(self, prompt_text: str, out_a: str, out_b: str) -> str:
        return (
            "You are an expert code reviewer and judge. Compare two candidate outputs for the same user prompt.\n\n"
            f"User prompt:\n{prompt_text}\n\n"
            f"Candidate A:\n{out_a}\n\n"
            f"Candidate B:\n{out_b}\n\n"
            "Task: Which candidate is better for correctness, clarity, and usefulness for a developer?\n\n"
            "Think through your reasoning, then end your response with EXACTLY one of these tokens:\n"
            "- A (if Candidate A is better)\n"
            "- B (if Candidate B is better)\n"
            "- TIE (if both are equally good)\n\n"
            "Your final answer must be just the single token A, B, or TIE."
        )

    def _extract_decision(self, response_text: str) -> Optional[str]:
        if not response_text:
            return None

        cleaned = response_text.strip().upper()

        # Strategy 1: Look for decision at the end
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            if last_line in ("A", "B", "TIE"):
                return last_line
            words = cleaned.split()
            for word in reversed(words[-5:]):
                if word in ("A", "B", "TIE"):
                    return word

        # Strategy 3: Pattern matching
        patterns = [
            r'\b(TIE)\b',
            r'(?:final|answer|decision|conclusion|choice).*?\b([AB])\b',
            r'\b([AB])\s*(?:is|wins|better)',
            r'(?:candidate\s+)?([AB])(?:\s+(?:is|wins))',
            r'^.*?([AB])',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, cleaned)
            if matches:
                return matches[-1]

        # Strategy 4: Tie indicators
        tie_keywords = ['EQUAL', 'SAME', 'TIED', 'DRAW', 'EQUIVALENT', 'BOTH ARE']
        if any(keyword in cleaned for keyword in tie_keywords):
            return "TIE"

        # Strategy 5: Count mentions of A vs B
        a_count = len(re.findall(r'\bA\b', cleaned))
        b_count = len(re.findall(r'\bB\b', cleaned))
        if a_count > b_count + 1:
            return "A"
        elif b_count > a_count + 1:
            return "B"

        return None

    def judge_once(
        self,
        prompt_text: str,
        out_a: str,
        out_b: str,
        judge_model_config: ModelCapability,
        timeout: int = 120,
    ) -> Dict:
        judge_prompt = self._create_judge_prompt(prompt_text, out_a, out_b)

        try:
            max_tokens = 32

            res = self.client_manager.call_with_fallback(
                judge_model_config,
                judge_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                timeout=timeout,
            )

            if not res.ok:
                return {
                    "ok": False,
                    "reason": f"API call failed: {res.error or 'Unknown error'}",
                    "raw": res,
                }

            response_text = res.text

            if not response_text and res.raw:
                raw_data = res.raw
                if hasattr(raw_data, "choices") and raw_data.choices:
                    choice = raw_data.choices[0]
                    message = getattr(choice, "message", None) or {}
                    response_text = getattr(message, "content", "") or ""
                    if not response_text:
                        reasoning = getattr(message, "reasoning", "") or ""
                        if reasoning:
                            words = reasoning.split()[-10:]
                            response_text = " ".join(words) if words else reasoning

            if not response_text:
                return {
                    "ok": False,
                    "reason": "Empty response from judge model",
                    "raw": res,
                }

            decision = self._extract_decision(response_text)

            if decision:
                return {"ok": True, "decision": decision, "raw": res}
            else:
                return {
                    "ok": False,
                    "reason": f"Could not parse decision from response: '{response_text}'",
                    "raw": res,
                }

        except Exception as e:
            return {"ok": False, "reason": f"Exception during judgment: {str(e)}", "raw": {}}

    def judge_majority(
        self,
        prompt_text: str,
        out_a: str,
        out_b: str,
        judge_model_config: ModelCapability,
        repeats: int = 1,
        timeout: int = 120,
    ) -> Dict:
        votes: List[str] = []
        raw_calls: List[Dict] = []
        errors: List[str] = []

        for attempt in range(repeats):
            try:
                jr = self.judge_once(
                    prompt_text, out_a, out_b, judge_model_config, timeout=timeout
                )
                raw_calls.append(jr)

                if jr.get("ok"):
                    votes.append(jr["decision"])
                else:
                    votes.append("ERR")
                    errors.append(
                        f"Attempt {attempt + 1}: {jr.get('reason', 'Unknown error')}"
                    )

            except Exception as e:
                votes.append("ERR")
                errors.append(f"Attempt {attempt + 1}: Exception - {str(e)}")
                raw_calls.append({"ok": False, "reason": str(e)})

        counts = {"A": 0, "B": 0, "TIE": 0, "ERR": 0}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1

        valid_votes = counts["A"] + counts["B"] + counts["TIE"]
        if valid_votes == 0:
            return {
                "ok": False,
                "reason": {
                    "message": "All judgment attempts failed",
                    "votes": votes,
                    "errors": errors,
                    "raw": raw_calls,
                },
            }

        valid_counts = {k: v for k, v in counts.items() if k != "ERR"}
        winner = max(valid_counts, key=lambda k: valid_counts[k])
        max_count = valid_counts[winner]
        tied_winners = [k for k, v in valid_counts.items() if v == max_count]
        final_decision = "TIE" if len(tied_winners) > 1 else winner

        result: Dict = {
            "ok": True,
            "decision": final_decision,
            "counts": counts,
            "votes": votes,
            "valid_votes": valid_votes,
            "total_attempts": repeats,
        }

        if errors:
            result["errors"] = errors
        if any(not call.get("ok", False) for call in raw_calls):
            result["raw"] = raw_calls

        return result
