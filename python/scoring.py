from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


class Scorer:
    name = "base"

    def score(self, text: str, rubric: dict[str, str]) -> dict[str, float]:
        raise NotImplementedError


class RubricScorer(Scorer):
    name = "rubric-evaluator"

    def score(self, text: str, rubric: dict[str, str]) -> dict[str, float]:
        n = len(text)
        clarity = min(95, 62 + min(n, 55) * 0.45)
        evidence_keys = ["因为", "例如", "证据", "理由", "数据", "所以"]
        interaction_keys = ["同意", "不过", "回应", "观点", "差异", "复核"]
        action_keys = ["建议", "可以", "最好", "设置", "只保存", "指标", "期限"]
        evidence = min(96, 60 + 8 * sum(k in text for k in evidence_keys))
        interaction = min(96, 58 + 8 * sum(k in text for k in interaction_keys))
        actionability = min(96, 58 + 8 * sum(k in text for k in action_keys))
        total = round(0.30 * clarity + 0.25 * evidence + 0.25 * interaction + 0.20 * actionability, 1)
        return {
            "clarity": round(clarity, 1),
            "evidence": float(evidence),
            "interaction": float(interaction),
            "actionability": float(actionability),
            "total": total,
        }


class OpenAICompatibleScorer(Scorer):
    name = "openai-compatible-llm"

    def __init__(self, endpoint: str, api_key: str, model: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model

    def score(self, text: str, rubric: dict[str, str]) -> dict[str, float]:
        prompt = (
            "请按 0-100 分评价课堂发言。维度为 clarity/evidence/interaction/actionability，"
            "并给出 total。只返回 JSON 对象。\n"
            f"评分标准：{json.dumps(rubric, ensure_ascii=False)}\n"
            f"发言：{text}"
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "你是课堂讨论质量评估器。"},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=45) as resp:
            data: dict[str, Any] = json.load(resp)
        content = data["choices"][0]["message"]["content"]
        result = json.loads(content)
        return {k: float(result[k]) for k in ["clarity", "evidence", "interaction", "actionability", "total"]}


def build_scorer() -> Scorer:
    endpoint = os.getenv("LLM_API_URL", "").strip()
    if endpoint:
        return OpenAICompatibleScorer(endpoint, os.getenv("LLM_API_KEY", ""), os.getenv("LLM_MODEL", "gpt-4o-mini"))
    return RubricScorer()
