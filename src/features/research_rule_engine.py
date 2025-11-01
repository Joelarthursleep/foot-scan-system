"""
Research-backed rule engine.
Converts structured clinical evidence into actionable checks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class ResearchRule:
    condition: str
    metric: str
    threshold: float
    operator: str
    cohort: Optional[Dict[str, Any]]
    citation: str
    confidence: float
    severity: str
    description: str

    def matches(self, value: float, context: Dict[str, Any]) -> bool:
        """Evaluate rule with optional cohort filtering."""
        if value is None:
            return False

        if self.cohort:
            for key, expected in self.cohort.items():
                if expected is None:
                    continue
                if context.get(key) != expected:
                    return False

        ops = {
            ">": value > self.threshold,
            ">=": value >= self.threshold,
            "<": value < self.threshold,
            "<=": value <= self.threshold,
            "==": value == self.threshold
        }
        return ops.get(self.operator, False)


class ResearchRuleEngine:
    """Loads and applies research-based thresholds."""

    def __init__(self, rule_file: Optional[str] = None):
        self.rules: Dict[str, List[ResearchRule]] = {}
        if rule_file:
            self.load_rules(Path(rule_file))
        else:
            default_path = Path("data/research_rules.json")
            if default_path.exists():
                self.load_rules(default_path)
            else:
                LOGGER.info("No research rule file found. Evidence-based augmentations disabled.")

    def load_rules(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text())
            for entry in payload:
                rule = ResearchRule(
                    condition=entry["condition"],
                    metric=entry["metric"],
                    threshold=float(entry["threshold"]),
                    operator=entry.get("operator", ">="),
                    cohort=entry.get("cohort"),
                    citation=entry.get("citation", ""),
                    confidence=float(entry.get("confidence", 0.5)),
                    severity=entry.get("severity", "moderate"),
                    description=entry.get("description", "")
                )
                self.rules.setdefault(rule.condition.lower(), []).append(rule)
            LOGGER.info("Loaded %d research rules from %s", len(payload), path)
        except Exception as exc:
            LOGGER.error("Failed to load research rules: %s", exc)

    def evaluate(self, metrics: Dict[str, float], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return triggered rules given measurement dictionary."""
        triggers: List[Dict[str, Any]] = []
        if not self.rules:
            return triggers

        for condition, rules in self.rules.items():
            for rule in rules:
                value = metrics.get(rule.metric)
                if value is None:
                    continue
                if rule.matches(value, context):
                    triggers.append({
                        "name": rule.condition,
                        "metric": rule.metric,
                        "value": value,
                        "threshold": rule.threshold,
                        "operator": rule.operator,
                        "severity": rule.severity,
                        "confidence": rule.confidence,
                        "citation": rule.citation,
                        "description": rule.description
                    })
        return triggers
