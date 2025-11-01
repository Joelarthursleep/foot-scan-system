"""
Risk matrix computation utilities.
"""
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class RiskMatrixResult:
    risk_tier: str
    score: float
    drivers: List[str]
    recommendation: str


class RiskMatrixBuilder:
    """
    Combine thresholds, ML outputs, and uncertainty into a single risk tier.
    """

    def build(self,
              health_score: float,
              high_significance_conditions: int,
              risk_assessments: List[Dict[str, Any]],
              uncertainty: float = 0.1) -> RiskMatrixResult:
        score = 0.0
        drivers: List[str] = []

        if health_score is not None:
            score += max(0, 100 - health_score) / 20
            if health_score < 60:
                drivers.append("Low foot health score")

        score += high_significance_conditions * 1.5
        if high_significance_conditions:
            drivers.append(f"{high_significance_conditions} high-significance conditions")

        for risk in risk_assessments:
            level = risk.get("risk_level", "").lower()
            probability = risk.get("probability", 0)
            if level == "high":
                score += 3 * probability
                drivers.append(f"High {risk.get('category', '')} risk")
            elif level == "medium":
                score += 1.5 * probability
                drivers.append(f"Medium {risk.get('category', '')} risk")

        score += uncertainty * 2

        if score >= 8:
            tier = "Critical"
            recommendation = "Immediate podiatry referral and fall risk mitigation."
        elif score >= 5:
            tier = "Elevated"
            recommendation = "Schedule comprehensive review within 4 weeks."
        elif score >= 2.5:
            tier = "Moderate"
            recommendation = "Increase monitoring cadence and reinforce preventive care."
        else:
            tier = "Low"
            recommendation = "Continue routine monitoring."

        return RiskMatrixResult(
            risk_tier=tier,
            score=round(score, 2),
            drivers=drivers[:5],
            recommendation=recommendation
        )
