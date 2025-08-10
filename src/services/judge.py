from typing import Dict, Optional, List
import re
import json
from .openrouter import OpenRouterClient

class JudgeService:
    def __init__(self, client: OpenRouterClient):
        self.client = client

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
        """Extract decision from response text with multiple fallback strategies"""
        if not response_text:
            return None
            
        # Clean the response
        cleaned = response_text.strip().upper()
        
        # Strategy 1: Look for decision at the end of the response (common for reasoning models)
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        if lines:
            # Check last line first
            last_line = lines[-1]
            if last_line in ("A", "B", "TIE"):
                return last_line
            
            # Check last few words
            words = cleaned.split()
            for word in reversed(words[-10:]):  # Check last 10 words
                if word in ("A", "B", "TIE"):
                    return word
        
        # Strategy 2: Direct match anywhere in text
        if "TIE" in cleaned:
            return "TIE"
        
        # Strategy 3: Pattern matching
        patterns = [
            r'\b(TIE)\b',
            r'(?:final|answer|decision|conclusion|choice).*?\b([AB])\b',
            r'\b([AB])\s*(?:is|wins|better)',
            r'(?:candidate\s+)?([AB])(?:\s+(?:is|wins))',
            r'([AB])(?:\s*$|\s*[.!])'  # A or B at end or before punctuation
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned)
            if matches:
                return matches[-1]  # Return the last match
        
        # Strategy 4: Look for tie indicators
        tie_keywords = ['EQUAL', 'SAME', 'TIED', 'DRAW', 'EQUIVALENT', 'BOTH ARE']
        if any(keyword in cleaned for keyword in tie_keywords):
            return "TIE"
        
        # Strategy 5: Count mentions of A vs B (fallback)
        a_count = len(re.findall(r'\bA\b', cleaned))
        b_count = len(re.findall(r'\bB\b', cleaned))
        
        if a_count > b_count + 1:  # Significantly more A mentions
            return "A"
        elif b_count > a_count + 1:  # Significantly more B mentions  
            return "B"
            
        return None

    def _extract_response_text(self, raw_response: Dict) -> str:
        """Extract text from API response, handling both regular and reasoning models"""
        
        # Try direct text field first
        text = raw_response.get("text", "")
        if text.strip():
            return text
        
        # For reasoning models, check the raw response structure
        raw_data = raw_response.get("raw", {})
        if not raw_data or "choices" not in raw_data:
            return ""
        
        choices = raw_data.get("choices", [])
        if not choices:
            return ""
        
        choice = choices[0]
        message = choice.get("message", {})
        
        # Try content field
        content = message.get("content", "")
        if content.strip():
            return content
        
        # For reasoning models like DeepSeek R1, check reasoning field
        reasoning = message.get("reasoning", "")
        if reasoning.strip():
            # For reasoning models, the actual decision is usually at the end
            # But we should return the full reasoning to extract the decision
            return reasoning
        
        return ""

    def _is_reasoning_model(self, model_id: str) -> bool:
        """Check if model is a reasoning model that needs special handling"""
        reasoning_keywords = ["deepseek-r1", "r1-", "reasoning", "think"]
        model_lower = model_id.lower()
        return any(keyword in model_lower for keyword in reasoning_keywords)

    def judge_once(self, prompt_text: str, out_a: str, out_b: str, judge_model_id: str, timeout: int = 120) -> Dict:
        """Single judgment with robust error handling"""
        judge_prompt = self._create_judge_prompt(prompt_text, out_a, out_b)
        
        try:
            # Determine appropriate token count based on model type
            if self._is_reasoning_model(judge_model_id):
                max_tokens = 800  # Much higher for reasoning models
            else:
                max_tokens = 150  # Conservative for regular models
            
            res = self.client.call(
                judge_model_id, 
                judge_prompt, 
                max_tokens=max_tokens,
                temperature=0.0, 
                top_p=1.0, 
                timeout=timeout
            )
            
            if not res.get("ok", False):
                return {
                    "ok": False, 
                    "reason": f"API call failed: {res.get('error', 'Unknown error')}", 
                    "raw": res
                }
            
            # Extract response text using improved method
            response_text = self._extract_response_text(res)
            
            if not response_text.strip():
                return {
                    "ok": False, 
                    "reason": "Empty response from judge model", 
                    "raw": res
                }
            
            # Try to extract decision
            decision = self._extract_decision(response_text)
            
            if decision:
                return {
                    "ok": True, 
                    "decision": decision, 
                    "raw": res,
                    "response_text": response_text[:200]  # For debugging
                }
            else:
                # Enhanced error message with actual response preview
                response_preview = response_text[:200] + ("..." if len(response_text) > 200 else "")
                return {
                    "ok": False, 
                    "reason": f"Could not parse decision from response: '{response_preview}'", 
                    "raw": res
                }
                
        except Exception as e:
            return {
                "ok": False, 
                "reason": f"Exception during judgment: {str(e)}", 
                "raw": {}
            }

    def judge_majority(
        self, 
        prompt_text: str, 
        out_a: str, 
        out_b: str, 
        judge_model_id: str, 
        repeats: int = 1, 
        timeout: int = 120
    ) -> Dict:
        """Run multiple judgments and return majority decision"""
        votes: List[str] = []
        raw_calls: List[Dict] = []
        errors: List[str] = []
        
        for attempt in range(repeats):
            try:
                jr = self.judge_once(prompt_text, out_a, out_b, judge_model_id, timeout=timeout)
                raw_calls.append(jr)
                
                if jr.get("ok"):
                    votes.append(jr["decision"])
                else:
                    votes.append("ERR")
                    errors.append(f"Attempt {attempt + 1}: {jr.get('reason', 'Unknown error')}")
                    
            except Exception as e:
                votes.append("ERR")
                errors.append(f"Attempt {attempt + 1}: Exception - {str(e)}")
                raw_calls.append({"ok": False, "reason": str(e)})
        
        # Count valid votes
        counts = {"A": 0, "B": 0, "TIE": 0, "ERR": 0}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        
        # Check if we have any valid votes
        valid_votes = counts["A"] + counts["B"] + counts["TIE"]
        if valid_votes == 0:
            return {
                "ok": False, 
                "reason": {
                    "message": "All judgment attempts failed",
                    "votes": votes, 
                    "errors": errors,
                    "raw": raw_calls
                }
            }
        
        # Find winner among valid votes only
        valid_counts = {k: v for k, v in counts.items() if k != "ERR"}
        winner = max(valid_counts, key=lambda k: valid_counts[k])
        
        # Check for ties among valid votes
        max_count = valid_counts[winner]
        tied_winners = [k for k, v in valid_counts.items() if v == max_count]
        
        final_decision = "TIE" if len(tied_winners) > 1 else winner
        
        result = {
            "ok": True, 
            "decision": final_decision, 
            "counts": counts, 
            "votes": votes, 
            "valid_votes": valid_votes,
            "total_attempts": repeats
        }
        
        # Include errors if any occurred
        if errors:
            result["errors"] = errors
            
        # Include raw data for debugging (optional)
        if any(not call.get("ok", False) for call in raw_calls):
            result["raw"] = raw_calls
            
        return result

    def debug_judge_response(self, prompt_text: str, out_a: str, out_b: str, judge_model_id: str, timeout: int = 120) -> Dict:
        """Debug method to see exactly what the judge is returning"""
        judge_prompt = self._create_judge_prompt(prompt_text, out_a, out_b)
        
        print(f"Judge prompt (first 200 chars): {judge_prompt[:200]}...")
        print(f"Using model: {judge_model_id}")
        
        max_tokens = 800 if self._is_reasoning_model(judge_model_id) else 150
        
        res = self.client.call(
            judge_model_id, 
            judge_prompt, 
            max_tokens=max_tokens,
            temperature=0.0, 
            top_p=1.0, 
            timeout=timeout
        )
        
        print(f"Raw response: {json.dumps(res, indent=2)}")
        
        response_text = self._extract_response_text(res)
        print(f"Extracted text: {response_text}")
        
        if response_text:
            decision = self._extract_decision(response_text)
            print(f"Extracted decision: {decision}")
        
        return res