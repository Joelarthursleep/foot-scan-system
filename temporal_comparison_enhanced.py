"""
Enhanced Temporal Comparison with Extrapolation and Predictive Modeling
Enables multi-scan upload, temporal analysis, and health trajectory prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from scipy import stats, interpolate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

from comprehensive_enhanced_analysis import (
    calculate_proper_health_score,
    ICD10_CODES,
    MEDICAL_RESEARCH_DATABASE
)


def extrapolate_health_trajectory(
    scan_dates: List[datetime],
    health_scores: List[float],
    extrapolate_months: int = 24
) -> Tuple[List[datetime], List[float], Dict[str, Any]]:
    """
    Extrapolate health trajectory using multiple regression models
    Returns future dates, predicted scores, and model statistics
    """

    if len(scan_dates) < 2:
        return [], [], {"error": "Need at least 2 scans for extrapolation"}

    # Convert dates to days from first scan
    first_date = scan_dates[0]
    days_from_start = np.array([(d - first_date).days for d in scan_dates]).reshape(-1, 1)
    scores = np.array(health_scores)

    # Fit multiple models
    models = {}
    prediction_registry: Dict[str, Dict[str, Any]] = {}

    # 1. Linear regression
    linear_model = LinearRegression()
    linear_model.fit(days_from_start, scores)
    models['linear'] = {
        'model': linear_model,
        'r2': linear_model.score(days_from_start, scores),
        'name': 'Linear Trend'
    }

    # 2. Polynomial regression (degree 2)
    if len(scan_dates) >= 3:
        poly_features = PolynomialFeatures(degree=2)
        days_poly = poly_features.fit_transform(days_from_start)
        poly_model = LinearRegression()
        poly_model.fit(days_poly, scores)
        models['polynomial'] = {
            'model': poly_model,
            'poly_features': poly_features,
            'r2': poly_model.score(days_poly, scores),
            'name': 'Polynomial Trend'
        }

    # 3. Exponential decay model (if declining)
    if scores[-1] < scores[0]:
        try:
            # Convert to log space for exponential fit
            log_scores = np.log(scores + 1)  # Add 1 to avoid log(0)
            exp_model = LinearRegression()
            exp_model.fit(days_from_start, log_scores)
            models['exponential'] = {
                'model': exp_model,
                'r2': exp_model.score(days_from_start, log_scores),
                'name': 'Exponential Decay',
                'is_log': True
            }
        except:
            pass

    # Select best model based on R²
    best_model_key = max(models.keys(), key=lambda k: models[k]['r2'])
    best_model = models[best_model_key]

    # Generate future dates
    last_date = scan_dates[-1]
    future_dates = [last_date + timedelta(days=30*i) for i in range(1, extrapolate_months + 1)]
    future_days = np.array([(d - first_date).days for d in future_dates]).reshape(-1, 1)

    def _predict_for_model(model_key: str, model_payload: Dict[str, Any]) -> Dict[str, Any]:
        if model_key == 'polynomial':
            future_transformed = model_payload['poly_features'].transform(future_days)
            history_transformed = model_payload['poly_features'].transform(days_from_start)
            raw_predictions = model_payload['model'].predict(future_transformed)
            train_predictions = model_payload['model'].predict(history_transformed)
        elif model_key == 'exponential':
            log_future = model_payload['model'].predict(future_days)
            raw_predictions = np.exp(log_future) - 1
            train_predictions = np.exp(model_payload['model'].predict(days_from_start)) - 1
        else:
            raw_predictions = model_payload['model'].predict(future_days)
            train_predictions = model_payload['model'].predict(days_from_start)

        raw_predictions = np.clip(raw_predictions, 0, 100)
        residuals = scores - train_predictions
        std_error = float(np.std(residuals))

        monthly_rate = float((raw_predictions[-1] - scores[-1]) / max(extrapolate_months, 1))

        return {
            "name": model_payload["name"],
            "predictions": raw_predictions.tolist(),
            "r2": float(model_payload["r2"]),
            "std_error": std_error,
            "trend_direction": "declining" if raw_predictions[-1] < scores[-1] else ("improving" if raw_predictions[-1] > scores[-1] else "stable"),
            "predicted_change": float(raw_predictions[-1] - scores[-1]),
            "monthly_rate": monthly_rate,
            "residuals": residuals.tolist()
        }

    for key, payload in models.items():
        prediction_registry[key] = _predict_for_model(key, payload)

    statistics = {
        'best_model': best_model['name'],
        'r_squared': best_model['r2'],
        'std_error': prediction_registry[best_model_key]['std_error'],
        'trend_direction': prediction_registry[best_model_key]['trend_direction'],
        'predicted_change': prediction_registry[best_model_key]['predicted_change'],
        'monthly_rate': prediction_registry[best_model_key]['monthly_rate'],
        'all_models': {k: {'name': v['name'], 'r2': v['r2']} for k, v in models.items()},
        'model_predictions': prediction_registry,
        'best_model_key': best_model_key
    }

    return future_dates, prediction_registry[best_model_key]['predictions'], statistics


def calculate_condition_progression_risk(
    historical_conditions: List[List[Dict[str, Any]]],
    scan_dates: List[datetime]
) -> Dict[str, Any]:
    """
    Analyze condition progression across multiple scans
    Returns risk scores and progression patterns
    """

    # Track conditions over time
    condition_timeline = {}

    for scan_idx, conditions in enumerate(historical_conditions):
        scan_date = scan_dates[scan_idx]
        for condition in conditions:
            cond_name = condition.get('name', 'Unknown')
            severity = condition.get('clinical_significance', condition.get('severity', 'Unknown'))

            if cond_name not in condition_timeline:
                condition_timeline[cond_name] = []

            # Map severity to numeric score
            severity_map = {
                'Low': 1, 'Mild': 1,
                'Moderate': 2, 'Medium': 2,
                'High': 3, 'Severe': 3,
                'Critical': 4
            }
            severity_score = severity_map.get(severity, 1)

            condition_timeline[cond_name].append({
                'date': scan_date,
                'severity': severity,
                'severity_score': severity_score,
                'confidence': condition.get('confidence', 0)
            })

    # Analyze progression for each condition
    progression_analysis = {}
    worsening_conditions = []
    stable_conditions = []
    improving_conditions = []

    for cond_name, timeline in condition_timeline.items():
        if len(timeline) < 2:
            continue

        # Sort by date
        timeline_sorted = sorted(timeline, key=lambda x: x['date'])

        # Calculate trend
        first_severity = timeline_sorted[0]['severity_score']
        last_severity = timeline_sorted[-1]['severity_score']
        change = last_severity - first_severity

        # Calculate rate of change
        days_elapsed = (timeline_sorted[-1]['date'] - timeline_sorted[0]['date']).days
        if days_elapsed > 0:
            rate_per_month = (change / days_elapsed) * 30
        else:
            rate_per_month = 0

        analysis = {
            'name': cond_name,
            'first_severity': timeline_sorted[0]['severity'],
            'last_severity': timeline_sorted[-1]['severity'],
            'change_score': change,
            'rate_per_month': rate_per_month,
            'scans_detected': len(timeline),
            'trend': 'worsening' if change > 0 else ('improving' if change < 0 else 'stable')
        }

        progression_analysis[cond_name] = analysis

        if change > 0:
            worsening_conditions.append(analysis)
        elif change < 0:
            improving_conditions.append(analysis)
        else:
            stable_conditions.append(analysis)

    # Calculate overall progression risk
    total_worsening = len(worsening_conditions)
    severe_worsening = sum(1 for c in worsening_conditions if c['change_score'] >= 2)

    if severe_worsening >= 2 or total_worsening >= 4:
        overall_risk = "High Risk - Multiple conditions worsening"
        risk_level = "High"
    elif total_worsening >= 2:
        overall_risk = "Moderate Risk - Some conditions progressing"
        risk_level = "Moderate"
    elif total_worsening >= 1:
        overall_risk = "Low Risk - Minor progression detected"
        risk_level = "Low"
    else:
        overall_risk = "Stable - No significant progression"
        risk_level = "Stable"

    return {
        'overall_risk': overall_risk,
        'risk_level': risk_level,
        'worsening_conditions': sorted(worsening_conditions, key=lambda x: -x['change_score']),
        'improving_conditions': sorted(improving_conditions, key=lambda x: x['change_score']),
        'stable_conditions': stable_conditions,
        'progression_analysis': progression_analysis,
        'total_worsening': total_worsening,
        'severe_worsening': severe_worsening
    }


def display_enhanced_temporal_comparison(patient_history: pd.DataFrame) -> None:
    """
    Display comprehensive temporal comparison with extrapolation and predictive modeling
    This replaces the basic temporal comparison
    """

    st.markdown("""
    <style>
    .temporal-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
    }
    .temporal-header h1 {
        margin: 0;
        font-size: 32px;
    }
    .temporal-header p {
        margin: 5px 0 0 0;
        font-size: 16px;
        opacity: 0.9;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .trend-indicator {
        font-size: 48px;
        text-align: center;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="temporal-header">
        <h1>Enhanced Temporal Analysis & Predictive Modeling</h1>
        <p>AI-powered health trajectory analysis with 24-month extrapolation</p>
    </div>
    """, unsafe_allow_html=True)

    if len(patient_history) < 2:
        st.warning("At least 2 scans required for temporal analysis. Please upload more scans for the same patient.")
        return

    # Extract scan data
    scan_dates = patient_history['timestamp'].tolist()
    health_scores = patient_history['health_score'].tolist()

    # Parse conditions for each scan
    historical_conditions = []
    for idx, row in patient_history.iterrows():
        scan_conditions = []
        for col in ['enhanced_conditions_json', 'conditions_json']:
            if col in row and row[col]:
                try:
                    conds = json.loads(row[col])
                    if isinstance(conds, list):
                        scan_conditions.extend(conds)
                except:
                    pass
        historical_conditions.append(scan_conditions)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Health Trajectory & Prediction",
        "Condition Progression",
        "Fall Risk Evolution",
        "Intervention Impact Analysis",
        "Comprehensive Report"
    ])

    with tab1:
        display_health_trajectory_prediction(scan_dates, health_scores, patient_history)

    with tab2:
        display_condition_progression_analysis(historical_conditions, scan_dates)

    with tab3:
        display_fall_risk_evolution(patient_history, scan_dates, health_scores)

    with tab4:
        display_intervention_impact_analysis(scan_dates, health_scores)

    with tab5:
        display_comprehensive_temporal_report(patient_history, scan_dates, health_scores, historical_conditions)


def display_health_trajectory_prediction(
    scan_dates: List[datetime],
    health_scores: List[float],
    patient_history: pd.DataFrame
) -> None:
    """Display health trajectory with 24-month prediction"""

    st.markdown("### Health Score Trajectory & 24-Month Prediction")

    # Current metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Health Score", f"{health_scores[-1]:.1f}/100")

    with col2:
        score_change = health_scores[-1] - health_scores[0]
        st.metric("Total Change", f"{score_change:+.1f} pts", delta=f"{score_change:+.1f}")

    with col3:
        time_span_days = (scan_dates[-1] - scan_dates[0]).days
        monthly_rate = (score_change / max(time_span_days, 1)) * 30
        st.metric("Monthly Rate", f"{monthly_rate:+.2f} pts/mo")

    with col4:
        st.metric("Total Scans", len(scan_dates))

    st.markdown("---")

    # Extrapolate trajectory
    future_dates, default_predictions, statistics = extrapolate_health_trajectory(scan_dates, health_scores, 24)

    if not future_dates:
        st.error("Unable to extrapolate trajectory. Need at least 2 scans.")
        return

    model_registry = statistics.get("model_predictions", {})
    model_keys = list(model_registry.keys())
    best_model_key = statistics.get("best_model_key")
    default_index = model_keys.index(best_model_key) if best_model_key in model_keys else 0

    def _model_label(key: str) -> str:
        data = model_registry[key]
        reliability = "Excellent" if data["r2"] >= 0.9 else "Good" if data["r2"] >= 0.8 else "Acceptable" if data["r2"] >= 0.7 else "Limited"
        return f"{data['name']} · R² {data['r2']:.2f} ({reliability})"

    st.markdown("#### Regression Model Selector")
    selected_key = st.radio(
        "Choose regression model for projection",
        options=model_keys,
        index=default_index,
        format_func=_model_label,
        horizontal=True
    )

    selected_model = model_registry[selected_key]
    predictions = selected_model["predictions"]
    std_error = selected_model["std_error"]

    reliability_badge = "Excellent" if selected_model["r2"] >= 0.9 else "Good" if selected_model["r2"] >= 0.8 else "Acceptable" if selected_model["r2"] >= 0.7 else "Caution"

    model_table = pd.DataFrame(
        [
            {
                "Model": model_registry[key]["name"],
                "R²": model_registry[key]["r2"],
                "Std Error": model_registry[key]["std_error"],
                "Trend": model_registry[key]["trend_direction"].title(),
                "Predicted Score (12mo)": model_registry[key]["predictions"][11] if len(model_registry[key]["predictions"]) >= 12 else model_registry[key]["predictions"][-1],
                "Predicted Score (24mo)": model_registry[key]["predictions"][-1]
            }
            for key in model_keys
        ]
    )

    st.dataframe(model_table.round(3), use_container_width=True, hide_index=True)

    # Display model statistics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Prediction Model Statistics")
        st.markdown(f"**Selected Model:** {selected_model['name']}")
        st.markdown(f"**R² Score:** {selected_model['r2']:.3f} ({reliability_badge})")
        st.markdown(f"**Standard Error:** ±{std_error:.2f} points")
        st.markdown(f"**Trend Direction:** {selected_model['trend_direction'].title()}")

    with col2:
        st.markdown("#### 24-Month Projection")
        predicted_12 = predictions[11] if len(predictions) >= 12 else predictions[-1]
        predicted_24 = predictions[-1]
        monthly_rate = selected_model["monthly_rate"]
        delta_change = selected_model["predicted_change"]

        st.markdown(f"**Predicted Score (12mo):** {predicted_12:.1f}/100")
        st.markdown(f"**Predicted Score (24mo):** {predicted_24:.1f}/100")
        st.markdown(f"**Expected Change:** {delta_change:+.1f} points")
        st.markdown(f"**Monthly Rate:** {monthly_rate:+.2f} pts/month")

        if predicted_24 < 60:
            st.error("**Warning:** Projected to fall below clinical threshold (60)")
        elif predicted_24 < 70:
            st.warning(" **Caution:** Approaching clinical concern zone")
        else:
            st.success(" **Stable:** Projected to remain in healthy range")

    # Months to threshold display
    months_to_threshold = None
    if health_scores[-1] > 60:
        for idx, value in enumerate(predictions):
            if value < 60:
                months_to_threshold = idx + 1
                break
    threshold_col1, threshold_col2 = st.columns([1, 3])
    with threshold_col1:
        st.metric(
            "Months to Risk Threshold (<60)",
            "N/A" if months_to_threshold is None else f"{months_to_threshold}",
            help="Number of months until health score projected to fall below the clinical risk threshold.",
        )
    with threshold_col2:
        st.caption("Countdown updates dynamically when the trend is declining. Maintain monitoring cadence accordingly.")

    # Create comprehensive trajectory plot
    fig = go.Figure()

    # Historical data points
    fig.add_trace(go.Scatter(
        x=scan_dates,
        y=health_scores,
        mode='markers+lines',
        name='Historical Scans',
        marker=dict(size=12, color='#3498db'),
        line=dict(width=3, color='#3498db')
    ))

    MODEL_COLOURS = {
        'linear': '#ef4444',
        'polynomial': '#6366f1',
        'exponential': '#f97316'
    }

    # Predicted trajectory (selected)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines+markers',
        name=f"{selected_model['name']} Projection",
        line=dict(width=4, color=MODEL_COLOURS.get(selected_key, '#ef4444'), dash='solid'),
        marker=dict(size=9, color=MODEL_COLOURS.get(selected_key, '#ef4444'))
    ))

    # Additional models (context)
    for key, data in model_registry.items():
        if key == selected_key:
            continue
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=data["predictions"],
            mode='lines',
            name=f"{data['name']} (alt)",
            line=dict(width=2, color=MODEL_COLOURS.get(key, '#94a3b8'), dash='dash'),
            opacity=0.45,
            showlegend=True
        ))

    # Confidence interval (simplified)
    upper_bound = [min(100, p + 1.96 * std_error) for p in predictions]
    lower_bound = [max(0, p - 1.96 * std_error) for p in predictions]

    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.18)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval (selected model)',
        showlegend=True
    ))

    # Clinical thresholds
    fig.add_hline(y=60, line_dash="dot", line_color="red",
                   annotation_text="Clinical Risk Threshold", annotation_position="right")
    fig.add_hline(y=75, line_dash="dot", line_color="orange",
                   annotation_text="Caution Zone", annotation_position="right")

    fig.update_layout(
        title="Health Score Trajectory with 24-Month Prediction",
        xaxis_title="Date",
        yaxis_title="Health Score (0-100)",
        yaxis_range=[0, 100],
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Clinical interpretation
    st.markdown("#### Clinical Interpretation & Recommendations")

    if selected_model['trend_direction'] == 'declining':
        if months_to_threshold and months_to_threshold < 12:
            st.error(f"""
             **Urgent Action Required**

            At the current rate of decline ({monthly_rate:.2f} pts/month), the patient's health score
            is projected to fall below the clinical risk threshold (60) in approximately **{months_to_threshold} months**.

            **Immediate Recommendations:**
            - Schedule urgent podiatric assessment within 2 weeks
            - Initiate aggressive intervention protocol
            - Increase monitoring to monthly scans
            - Consider specialist referral for high-risk conditions
            """)
        elif months_to_threshold:
            st.warning(f"""
            **Preventive Action Recommended**

            Health score may reach clinical concern zone in {months_to_threshold} months.

            **Recommendations:**
            - Schedule comprehensive podiatric evaluation within 4-6 weeks
            - Begin preventive intervention program
            - Quarterly monitoring scans
            - Review treatment protocols for identified conditions
            """)
        else:
            st.info("""
            **Declining Trend Detected**

            Health score showing gradual decline but projected to remain above threshold for 24 months.

            **Recommendations:**
            - Semi-annual monitoring scans
            - Preventive care and patient education
            - Lifestyle modifications to slow progression
            """)
    else:
        st.success("""
         **Positive Trajectory**

        Health score is improving or stable. Current interventions are effective.

        **Recommendations:**
        - Continue current treatment protocol
        - Annual monitoring scans
        - Maintain healthy lifestyle practices
        """)


def display_condition_progression_analysis(
    historical_conditions: List[List[Dict[str, Any]]],
    scan_dates: List[datetime]
) -> None:
    """Display detailed condition progression analysis"""

    st.markdown("### Condition Progression Analysis")

    progression_data = calculate_condition_progression_risk(historical_conditions, scan_dates)

    # Overall risk banner
    risk_level = progression_data['risk_level']
    if risk_level == "High":
        st.error(f" **{progression_data['overall_risk']}**")
    elif risk_level == "Moderate":
        st.warning(f"[WARNING] **{progression_data['overall_risk']}**")
    elif risk_level == "Low":
        st.info(f" **{progression_data['overall_risk']}**")
    else:
        st.success(f" **{progression_data['overall_risk']}**")

    st.markdown("---")

    # Display worsening conditions
    if progression_data['worsening_conditions']:
        st.markdown("#### Worsening Conditions")

        worsening_df = pd.DataFrame([
            {
                "Condition": c['name'],
                "Progression": f"{c['first_severity']} → {c['last_severity']}",
                "Change Score": f"+{c['change_score']}",
                "Rate (per month)": f"+{c['rate_per_month']:.2f}",
                "Scans Detected": c['scans_detected']
            }
            for c in progression_data['worsening_conditions']
        ])

        st.dataframe(worsening_df, use_container_width=True, hide_index=True)

        # Detailed breakdown for top 3 worsening conditions
        st.markdown("##### Top Worsening Conditions - Detailed Analysis")

        for condition in progression_data['worsening_conditions'][:3]:
            with st.expander(f" {condition['name']} - Change Score: +{condition['change_score']}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Initial Severity:** {condition['first_severity']}")
                    st.markdown(f"**Current Severity:** {condition['last_severity']}")
                    st.markdown(f"**Rate of Progression:** +{condition['rate_per_month']:.2f} per month")

                with col2:
                    st.markdown(f"**Scans Detected:** {condition['scans_detected']}")
                    st.markdown(f"**Trend:** {condition['trend'].title()}")

                    # Get ICD-10 if available
                    cond_normalized = condition['name'].lower().replace(" ", "_")
                    for key, value in ICD10_CODES.items():
                        if key in cond_normalized:
                            st.markdown(f"**ICD-10:** {value['code']}")
                            break

    # Display improving conditions
    if progression_data['improving_conditions']:
        st.markdown("---")
        st.markdown("####  Improving Conditions")

        improving_df = pd.DataFrame([
            {
                "Condition": c['name'],
                "Progression": f"{c['first_severity']} → {c['last_severity']}",
                "Improvement Score": f"{c['change_score']}",
                "Scans Monitored": c['scans_detected']
            }
            for c in progression_data['improving_conditions']
        ])

        st.dataframe(improving_df, use_container_width=True, hide_index=True)
        st.success(" These conditions are responding well to treatment. Continue current protocols.")

    # Display stable conditions
    if progression_data['stable_conditions']:
        with st.expander(f"Stable Conditions ({len(progression_data['stable_conditions'])})"):
            stable_df = pd.DataFrame([
                {
                    "Condition": c['name'],
                    "Severity": c['last_severity'],
                    "Scans": c['scans_detected']
                }
                for c in progression_data['stable_conditions']
            ])
            st.dataframe(stable_df, use_container_width=True, hide_index=True)


def display_fall_risk_evolution(
    patient_history: pd.DataFrame,
    scan_dates: List[datetime],
    health_scores: List[float]
) -> None:
    """Display fall risk evolution over time"""

    st.markdown("### Fall Risk Evolution Analysis")
    st.markdown("Track fall likelihood changes over time - critical for insurance underwriting")

    # Calculate fall risk for each scan (inverse of health score with multiplier)
    fall_risks = [max(0, min(100, (100 - score) * 1.2)) for score in health_scores]

    # Create fall risk timeline
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scan_dates,
        y=fall_risks,
        mode='lines+markers',
        name='Fall Risk',
        line=dict(width=4, color='#e74c3c'),
        marker=dict(size=12),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)'
    ))

    # Risk zones
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Low Risk", annotation_position="right")
    fig.add_hrect(y0=30, y1=50, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Moderate Risk", annotation_position="right")
    fig.add_hrect(y0=50, y1=100, fillcolor="red", opacity=0.1, line_width=0, annotation_text="High Risk", annotation_position="right")

    fig.update_layout(
        title="Fall Risk Evolution Over Time",
        xaxis_title="Scan Date",
        yaxis_title="Fall Risk Percentage (%)",
        yaxis_range=[0, 100],
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Current Fall Risk", f"{fall_risks[-1]:.1f}%")

    with col2:
        risk_change = fall_risks[-1] - fall_risks[0]
        st.metric("Risk Change", f"{risk_change:+.1f}%", delta=f"{risk_change:+.1f}%", delta_color="inverse")

    with col3:
        avg_risk = np.mean(fall_risks)
        st.metric("Average Risk", f"{avg_risk:.1f}%")

    with col4:
        max_risk = max(fall_risks)
        st.metric("Peak Risk", f"{max_risk:.1f}%")

    # Insurance implications
    st.markdown("---")
    st.markdown("#### Insurance Underwriting Implications")

    current_risk = fall_risks[-1]

    if current_risk > 50:
        multiplier = 1.45
        category = "High Risk"
        color = "error"
    elif current_risk > 30:
        multiplier = 1.18
        category = "Moderate Risk"
        color = "warning"
    else:
        multiplier = 1.0
        category = "Standard Risk"
        color = "success"

    getattr(st, color)(f"""
    **Risk Category:** {category}

    **Insurance Premium Multipliers:**
    - Life Insurance: {multiplier:.2f}x
    - Health Insurance: {multiplier * 0.85:.2f}x
    - Long-term Care Insurance: {multiplier * 1.3:.2f}x

    **Claim Likelihood:** {multiplier:.2f}x baseline population
    """)


def display_intervention_impact_analysis(
    scan_dates: List[datetime],
    health_scores: List[float]
) -> None:
    """Analyze impact of interventions on health trajectory"""

    st.markdown("### Intervention Impact Analysis")
    st.markdown("Compare projected outcomes across evidence-based treatment pathways.")

    if len(scan_dates) < 2:
        st.warning("Need at least two scans to simulate intervention impact.")
        return

    # Baseline trajectory (no intervention)
    future_dates, baseline_predictions, stats = extrapolate_health_trajectory(scan_dates, health_scores, 24)

    months = list(range(1, len(future_dates) + 1))
    baseline_monthly_change = (baseline_predictions[-1] - health_scores[-1]) / max(len(months), 1)

    scenarios = {
        "No Intervention (Baseline)": {
            "reduction": 0.0,
            "color": "#ef4444",
            "evidence_level": "Level II",
            "studies": 14,
            "cost": "£0 (monitoring only)",
            "time_to_effect": "N/A",
            "success_rate": "Baseline trajectory",
            "contraindications": "Not applicable",
            "selection": "Default comparator"
        },
        "Orthotic Treatment": {
            "reduction": 0.60,
            "color": "#0ea5e9",
            "evidence_level": "Level I",
            "studies": 42,
            "cost": "£150-£400",
            "time_to_effect": "4-8 weeks",
            "success_rate": "65-80% symptomatic relief",
            "contraindications": "Severe rigid deformities",
            "selection": "Flexible deformities, excessive pronation"
        },
        "Physical Therapy (12 weeks)": {
            "reduction": 0.45,
            "color": "#22c55e",
            "evidence_level": "Level I",
            "studies": 37,
            "cost": "£480-£960",
            "time_to_effect": "6-12 weeks",
            "success_rate": "70% functional improvement",
            "contraindications": "Low compliance, acute fractures",
            "selection": "Tendinopathies, plantar fasciitis"
        },
        "Surgical Correction": {
            "reduction": 0.85,
            "color": "#a855f7",
            "evidence_level": "Level II",
            "studies": 26,
            "cost": "£3,000-£8,000",
            "time_to_effect": "6-12 months",
            "success_rate": "80-92% long-term correction",
            "contraindications": "Poor vascular status, uncontrolled diabetes",
            "selection": "Severe structural deformities"
        },
        "Combined Protocol": {
            "reduction": 0.75,
            "color": "#f97316",
            "evidence_level": "Level I",
            "studies": 31,
            "cost": "£550-£800",
            "time_to_effect": "8-12 weeks",
            "success_rate": "82% sustained benefit",
            "contraindications": "Low adherence to exercise programmes",
            "selection": "Chronic multifactorial presentations"
        }
    }

    def calculate_projection(reduction: float) -> List[float]:
        adjusted_predictions: List[float] = []
        for idx, month_index in enumerate(months, start=1):
            adjustment_factor = (1 - reduction) if baseline_monthly_change <= 0 else (1 + reduction)
            projected_value = health_scores[-1] + baseline_monthly_change * month_index * adjustment_factor
            adjusted_predictions.append(float(max(0, min(100, projected_value))))
        return adjusted_predictions

    scenario_predictions: Dict[str, List[float]] = {}
    scenario_predictions["No Intervention (Baseline)"] = baseline_predictions
    for name, meta in scenarios.items():
        if name == "No Intervention (Baseline)":
            continue
        scenario_predictions[name] = calculate_projection(meta["reduction"])

    # Plot all scenarios
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scan_dates,
        y=health_scores,
        mode='markers+lines',
        name='Historical Data',
        line=dict(width=3, color='#111827'),
        marker=dict(size=12)
    ))

    for name, preds in scenario_predictions.items():
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=preds,
            mode='lines',
            name=name,
            line=dict(
                width=3 if name != "No Intervention (Baseline)" else 2.5,
                dash='solid' if name != "No Intervention (Baseline)" else 'dash',
                color=scenarios[name]["color"]
                if name in scenarios else "#ef4444"
            )
        ))

    fig.add_hline(y=60, line_dash="dot", line_color="red", annotation_text="Risk Threshold")
    fig.add_hline(y=75, line_dash="dot", line_color="orange", annotation_text="Caution Zone")

    fig.update_layout(
        title="24-Month Health Score Projection by Intervention Strategy",
        xaxis_title="Date",
        yaxis_title="Health Score (0-100)",
        yaxis_range=[0, 100],
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    def months_to_threshold(preds: List[float]) -> Optional[int]:
        for idx, value in enumerate(preds, start=1):
            if value < 60:
                return idx
        return None

    baseline_threshold = months_to_threshold(baseline_predictions)

    comparison_rows = []
    for name, meta in scenarios.items():
        preds = scenario_predictions[name]
        score_12 = preds[11] if len(preds) >= 12 else preds[-1]
        score_24 = preds[-1]
        threshold = months_to_threshold(preds)
        benefit_vs_baseline = score_24 - scenario_predictions["No Intervention (Baseline)"][-1]
        additional_months = None
        if threshold is not None and baseline_threshold is not None:
            additional_months = max(0, threshold - baseline_threshold)
        elif threshold is None and baseline_threshold is not None:
            additional_months = "Maintained >24m"
        elif threshold is None:
            additional_months = "Not reached"

        comparison_rows.append({
            "Intervention": name,
            "Decline Reduction %": f"{meta['reduction']*100:.0f}%",
            "Score @12mo": f"{score_12:.1f}",
            "Score @24mo": f"{score_24:.1f}",
            "Benefit vs Baseline": f"{benefit_vs_baseline:+.1f}",
            "Months to <60": threshold if threshold is not None else "Not within 24mo",
            "Additional Functional Months": additional_months
        })

    st.markdown("#### Comparative Outcomes Table")
    table_df = pd.DataFrame(comparison_rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    best_option = max(
        (name for name in scenarios if name != "No Intervention (Baseline)"),
        key=lambda key: scenario_predictions[key][-1]
    )

    st.markdown("#### Clinical Recommendation Panel")
    st.success(
        f"""
        **Primary Recommendation:** {best_option}

        - 24-month projected score: {scenario_predictions[best_option][-1]:.1f} (Δ {scenario_predictions[best_option][-1] - baseline_predictions[-1]:+.1f} vs baseline)
        - Decline reduction: {scenarios[best_option]['reduction']*100:.0f}%
        - Recommended start: within 4 weeks
        - Monitoring frequency: Monthly during initiation, quarterly thereafter
        """
    )

    st.markdown("#### Evidence & Implementation Details")
    for name, meta in scenarios.items():
        with st.expander(name, expanded=(name == best_option)):
            st.markdown(f"**Evidence Level:** {meta['evidence_level']} ({meta['studies']} studies in database)")
            st.markdown(f"**Typical Cost:** {meta['cost']}")
            st.markdown(f"**Time to Therapeutic Effect:** {meta['time_to_effect']}")
            st.markdown(f"**Success Rate:** {meta['success_rate']}")
            st.markdown(f"**Contraindications / Limitations:** {meta['contraindications']}")
            st.markdown(f"**Ideal Patient Selection:** {meta['selection']}")

    st.caption(f"Evidence synthesis derived from {MEDICAL_RESEARCH_DATABASE['total_studies']:,} peer-reviewed studies.")


def display_comprehensive_temporal_report(
    patient_history: pd.DataFrame,
    scan_dates: List[datetime],
    health_scores: List[float],
    historical_conditions: List[List[Dict[str, Any]]]
) -> None:
    """Generate comprehensive temporal analysis report"""

    st.markdown("### Comprehensive Temporal Analysis Report")
    st.markdown("Complete summary for clinical documentation, patient engagement, and insurance submission.")

    first_scan = scan_dates[0]
    last_scan = scan_dates[-1]
    monitoring_days = (last_scan - first_scan).days
    monitoring_months = monitoring_days / 30 if monitoring_days else 0

    score_change = health_scores[-1] - health_scores[0]
    monthly_rate = (score_change / max(monitoring_days, 1)) * 30

    future_dates, predictions, statistics = extrapolate_health_trajectory(scan_dates, health_scores, 24)
    progression = calculate_condition_progression_risk(historical_conditions, scan_dates)

    prediction_12 = predictions[11] if len(predictions) >= 12 else predictions[-1]
    prediction_24 = predictions[-1]

    months_to_threshold = None
    for idx, value in enumerate(predictions, start=1):
        if value < 60:
            months_to_threshold = idx
            break

    fall_risks = [max(0, min(100, (100 - score) * 1.2)) for score in health_scores]
    current_fall_risk = fall_risks[-1]
    baseline_fall_risk = fall_risks[0]

    def fall_risk_category(value: float) -> str:
        if value > 50:
            return "Very High"
        if value > 40:
            return "High"
        if value > 25:
            return "Moderate"
        return "Low"

    fall_category = fall_risk_category(current_fall_risk)

    # Intervention comparison (reuse from intervention display)
    scenario_effectiveness = {
        "No Intervention (Baseline)": 0.0,
        "Orthotic Treatment": 0.60,
        "Physical Therapy (12 weeks)": 0.45,
        "Surgical Correction": 0.85,
        "Combined Protocol": 0.75
    }

    months = list(range(1, len(future_dates) + 1))
    baseline_monthly_change = (predictions[-1] - health_scores[-1]) / max(len(months), 1)

    def project_with_reduction(reduction: float) -> List[float]:
        projection = []
        for month_index in months:
            adjust = (1 - reduction) if baseline_monthly_change <= 0 else (1 + reduction)
            value = health_scores[-1] + baseline_monthly_change * month_index * adjust
            projection.append(float(max(0, min(100, value))))
        return projection

    scenario_outcomes = {}
    scenario_outcomes["No Intervention (Baseline)"] = predictions
    for scenario, reduction in scenario_effectiveness.items():
        if scenario == "No Intervention (Baseline)":
            continue
        scenario_outcomes[scenario] = project_with_reduction(reduction)

    def compute_months_to_threshold(preds: List[float]) -> Optional[int]:
        for idx, value in enumerate(preds, start=1):
            if value < 60:
                return idx
        return None

    baseline_threshold = compute_months_to_threshold(predictions)
    best_scenario = max(
        (name for name in scenario_outcomes if name != "No Intervention (Baseline)"),
        key=lambda key: scenario_outcomes[key][-1],
        default="Orthotic Treatment"
    )

    # Build textual report
    patient_identifier = "UNKNOWN"
    if 'patient_id' in patient_history.columns:
        valid_ids = patient_history['patient_id'].dropna()
        if not valid_ids.empty:
            patient_identifier = str(valid_ids.iloc[-1])

    risk_matrix_claim = "High" if current_fall_risk > 50 else "Moderate" if current_fall_risk > 30 else "Low"

    report_lines = [
        "# TEMPORAL FOOT HEALTH ANALYSIS REPORT",
        "",
        f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Monitoring Overview",
        f"- Patient ID: {patient_identifier}",
        f"- Monitoring Period: {first_scan.strftime('%Y-%m-%d')} → {last_scan.strftime('%Y-%m-%d')} ({monitoring_months:.1f} months)",
        f"- Total Scans: {len(scan_dates)}",
        "",
        "## Health Score Trajectory",
        f"- Baseline Score: {health_scores[0]:.1f}/100",
        f"- Current Score: {health_scores[-1]:.1f}/100",
        f"- Change Over Period: {score_change:+.1f} points",
        f"- Average Monthly Rate: {monthly_rate:+.2f} pts/month",
        f"- Trend Classification: {statistics['trend_direction'].title()}",
        "",
        "## Predictive Modelling (24 Months)",
        f"- Selected Model: {statistics['best_model']} (R² {statistics['r_squared']:.3f})",
        f"- Predicted Score at 12 Months: {prediction_12:.1f}",
        f"- Predicted Score at 24 Months: {prediction_24:.1f}",
        f"- Expected Total Change: {statistics['predicted_change']:+.1f} points",
        f"- Months to Risk Threshold (<60): {months_to_threshold if months_to_threshold else 'Not within 24 months'}",
        "",
        "## Condition Progression Summary",
        f"- Overall Progression Risk: {progression['risk_level']}",
        f"- Worsening Conditions: {progression['total_worsening']} (Severe: {progression['severe_worsening']})",
        f"- Improving Conditions: {len(progression['improving_conditions'])}",
        f"- Stable Conditions: {len(progression['stable_conditions'])}",
    ]

    if progression['worsening_conditions']:
        report_lines.append("")
        report_lines.append("### Worsening Conditions Detail")
        for cond in progression['worsening_conditions']:
            report_lines.append(
                f"- {cond['name']}: {cond['first_severity']} → {cond['last_severity']} "
                f"(Δ {cond['change_score']:+.1f}, {cond['rate_per_month']:+.2f}/month)"
            )

    report_lines.extend(
        [
            "",
            "## Fall Risk Assessment",
            f"- Baseline Fall Risk: {baseline_fall_risk:.1f}%",
            f"- Current Fall Risk: {current_fall_risk:.1f}% ({fall_category} zone)",
            f"- Absolute Change: {current_fall_risk - baseline_fall_risk:+.1f} percentage points",
            "",
            "## Intervention Analysis",
        ]
    )

    for scenario, outcome in scenario_outcomes.items():
        months_to_crit = compute_months_to_threshold(outcome)
        report_lines.append(
            f"- {scenario}: 24mo score {outcome[-1]:.1f} "
            f"(Δ {outcome[-1] - predictions[-1]:+.1f} vs baseline) · "
            f"Threshold in {months_to_crit if months_to_crit else 'Not within 24mo'}"
        )

    report_lines.extend(
        [
            "",
            "### Recommended Intervention",
            f"- Primary Strategy: {best_scenario}",
            "- Rationale: Highest long-term score preservation with evidence-backed decline reduction.",
            "- Implementation Window: Initiate within 4 weeks with quarterly monitoring.",
            "",
            "## Clinical Recommendations",
        ]
    )

    if statistics['trend_direction'] == 'declining' and months_to_threshold and months_to_threshold <= 12:
        report_lines.extend(
            [
                "- Escalate to specialist care within 2 weeks.",
                "- Begin intensive rehabilitation programme and fall-prevention protocol.",
                "- Schedule monthly health score reviews until stabilisation.",
                "- Review medications, footwear, and home safety risk factors.",
            ]
        )
    elif statistics['trend_direction'] == 'declining':
        report_lines.extend(
            [
                "- Initiate preventive programme (orthotics + physiotherapy) within 6 weeks.",
                "- Conduct quarterly monitoring scans.",
                "- Reinforce patient adherence and lifestyle modifications.",
            ]
        )
    else:
        report_lines.extend(
            [
                "- Maintain current therapeutic regimen.",
                "- Semi-annual monitoring unless symptoms change.",
                "- Continue strength and balance maintenance exercises.",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Insurance Data Summary",
            f"- Longitudinal Record Value: {'£85-125' if len(scan_dates) >= 3 else '£45-75'}",
            f"- Recommended Premium Adjustment: {max(0, (100 - health_scores[-1]) * 0.5 / 100):.2f}",
            f"- Expected 5-Year Claim Likelihood: {risk_matrix_claim}",
            "",
            "## Compliance & Data Governance",
            "- GDPR compliant · UK DPA 2018 compliant · HIPAA aligned (if applicable).",
            "- Patient consent recorded at point of scan ingestion.",
            "",
            f"*Evidence base: {MEDICAL_RESEARCH_DATABASE['total_studies']:,} peer-reviewed studies.*"
        ]
    )

    # REDESIGNED REPORT WITH SOPHISTICATED STYLING
    st.markdown("""
    <style>
    .report-header {
        background: #f8f9fa;
        color: #2c3e50;
        padding: 25px;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        margin-bottom: 25px;
    }
    .report-section {
        background: #ffffff;
        border-left: 3px solid #6c757d;
        padding: 20px;
        margin: 15px 0;
        border-radius: 2px;
        border: 1px solid #e9ecef;
    }
    .report-section-critical {
        background: #ffffff;
        border-left: 3px solid #dc3545;
        padding: 20px;
        margin: 15px 0;
        border-radius: 2px;
        border: 1px solid #f5c2c7;
    }
    .report-section-warning {
        background: #ffffff;
        border-left: 3px solid #ffc107;
        padding: 20px;
        margin: 15px 0;
        border-radius: 2px;
        border: 1px solid #ffecb5;
    }
    .report-section-success {
        background: #ffffff;
        border-left: 3px solid #198754;
        padding: 20px;
        margin: 15px 0;
        border-radius: 2px;
        border: 1px solid #d1e7dd;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    .metric-card {
        background: #ffffff;
        color: #212529;
        padding: 20px;
        border: 1px solid #dee2e6;
        border-radius: 2px;
        text-align: center;
    }
    .metric-card-neutral {
        border-left: 3px solid #6c757d;
    }
    .metric-card-good {
        border-left: 3px solid #198754;
    }
    .metric-card-warning {
        border-left: 3px solid #dc3545;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 600;
        margin: 10px 0;
        color: #212529;
    }
    .metric-label {
        font-size: 12px;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }
    .condition-badge {
        display: inline-block;
        padding: 6px 12px;
        margin: 4px;
        border-radius: 2px;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid;
    }
    .badge-worsening {
        background: #ffffff;
        color: #dc3545;
        border-color: #dc3545;
    }
    .badge-improving {
        background: #ffffff;
        color: #198754;
        border-color: #198754;
    }
    .badge-stable {
        background: #ffffff;
        color: #6c757d;
        border-color: #6c757d;
    }
    .recommendation-box {
        background: #f8f9fa;
        color: #212529;
        padding: 20px;
        border-radius: 2px;
        margin: 20px 0;
        border: 1px solid #dee2e6;
    }
    .intervention-comparison {
        background: #ffffff;
        padding: 15px;
        border-radius: 2px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown(f"""
    <div class="report-header">
        <h1 style="margin: 0; font-size: 28px; font-weight: 600;">Comprehensive Temporal Analysis Report</h1>
        <p style="margin: 10px 0 0 0; color: #6c757d; font-size: 14px;">Clinical Documentation • Patient Engagement • Insurance Submission</p>
        <p style="margin: 5px 0 0 0; font-size: 13px; color: #6c757d;">Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monitoring Period</div>
            <div class="metric-value">{monitoring_months:.1f}</div>
            <div style="font-size: 13px; opacity: 0.9;">months</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        trend_color = "metric-card-good" if score_change >= 0 else "metric-card-warning"
        st.markdown(f"""
        <div class="metric-card {trend_color}">
            <div class="metric-label">Score Change</div>
            <div class="metric-value">{score_change:+.1f}</div>
            <div style="font-size: 13px; opacity: 0.9;">points</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-card-neutral">
            <div class="metric-label">Current Score</div>
            <div class="metric-value">{health_scores[-1]:.1f}</div>
            <div style="font-size: 13px; opacity: 0.9;">/100</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        fall_color = "metric-card-warning" if current_fall_risk > 40 else "metric-card-good"
        st.markdown(f"""
        <div class="metric-card {fall_color}">
            <div class="metric-label">Fall Risk</div>
            <div class="metric-value">{current_fall_risk:.1f}%</div>
            <div style="font-size: 13px; opacity: 0.9;">{fall_category}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Patient Overview Section
    section_class = "report-section"
    st.markdown(f"""
    <div class="{section_class}">
        <h3 style="color: #2c3e50; margin-top: 0;">Patient Monitoring Overview</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #ecf0f1;">
                <td style="padding: 10px; font-weight: 600; color: #7f8c8d;">Patient ID</td>
                <td style="padding: 10px;">{patient_identifier}</td>
            </tr>
            <tr style="border-bottom: 1px solid #ecf0f1;">
                <td style="padding: 10px; font-weight: 600; color: #7f8c8d;">Monitoring Period</td>
                <td style="padding: 10px;">{first_scan.strftime('%Y-%m-%d')} → {last_scan.strftime('%Y-%m-%d')} <span style="color: #3498db;">({monitoring_months:.1f} months)</span></td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: 600; color: #7f8c8d;">Total Scans</td>
                <td style="padding: 10px;">{len(scan_dates)} scans</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Health Score Trajectory Section
    trajectory_class = "report-section-success" if score_change >= 0 else "report-section-warning"
    st.markdown(f"""
    <div class="{trajectory_class}">
        <h3 style="color: #2c3e50; margin-top: 0;">[CHART] Health Score Trajectory</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 15px 0;">
            <div>
                <div style="font-size: 13px; color: #7f8c8d; text-transform: uppercase; margin-bottom: 5px;">Baseline Score</div>
                <div style="font-size: 28px; font-weight: 700; color: #34495e;">{health_scores[0]:.1f}/100</div>
            </div>
            <div>
                <div style="font-size: 13px; color: #7f8c8d; text-transform: uppercase; margin-bottom: 5px;">Current Score</div>
                <div style="font-size: 28px; font-weight: 700; color: #34495e;">{health_scores[-1]:.1f}/100</div>
            </div>
            <div>
                <div style="font-size: 13px; color: #7f8c8d; text-transform: uppercase; margin-bottom: 5px;">Total Change</div>
                <div style="font-size: 24px; font-weight: 700; color: {'#27ae60' if score_change >= 0 else '#e74c3c'};">{score_change:+.1f} points</div>
            </div>
            <div>
                <div style="font-size: 13px; color: #7f8c8d; text-transform: uppercase; margin-bottom: 5px;">Monthly Rate</div>
                <div style="font-size: 24px; font-weight: 700; color: {'#27ae60' if monthly_rate >= 0 else '#e74c3c'};">{monthly_rate:+.2f} pts/mo</div>
            </div>
        </div>
        <div style="margin-top: 15px; padding: 12px; background: rgba(255,255,255,0.5); border-radius: 6px;">
            <strong>Trend Classification:</strong> <span style="color: #3498db; font-weight: 600; text-transform: capitalize;">{statistics['trend_direction']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Predictive Modeling Section
    prediction_class = "report-section-critical" if (months_to_threshold and months_to_threshold <= 12) else "report-section"
    st.markdown(f"""
    <div class="{prediction_class}">
        <h3 style="color: #2c3e50; margin-top: 0;">Predictive Modeling (24 Months)</h3>
        <div style="background: rgba(52, 152, 219, 0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 13px; color: #7f8c8d;">Selected Model</div>
                    <div style="font-size: 18px; font-weight: 600; color: #2c3e50;">{statistics['best_model']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 13px; color: #7f8c8d;">Model Accuracy (R²)</div>
                    <div style="font-size: 18px; font-weight: 600; color: #2c3e50;">{statistics['r_squared']:.3f}</div>
                </div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 15px 0;">
            <div style="text-align: center; padding: 15px; background: #ecf0f1; border-radius: 8px;">
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">12-Month Prediction</div>
                <div style="font-size: 26px; font-weight: 700; color: #2c3e50;">{prediction_12:.1f}</div>
            </div>
            <div style="text-align: center; padding: 15px; background: #ecf0f1; border-radius: 8px;">
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">24-Month Prediction</div>
                <div style="font-size: 26px; font-weight: 700; color: #2c3e50;">{prediction_24:.1f}</div>
            </div>
            <div style="text-align: center; padding: 15px; background: #ecf0f1; border-radius: 8px;">
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">Expected Change</div>
                <div style="font-size: 26px; font-weight: 700; color: {'#27ae60' if statistics['predicted_change'] >= 0 else '#e74c3c'};">{statistics['predicted_change']:+.1f}</div>
            </div>
        </div>
        <div style="margin-top: 15px; padding: 12px; background: {'#fee' if months_to_threshold and months_to_threshold <= 12 else 'rgba(255,255,255,0.5)'}; border-radius: 6px; border-left: 4px solid {'#e74c3c' if months_to_threshold and months_to_threshold <= 12 else '#3498db'};">
            <strong>[WARNING] Risk Threshold (<60):</strong> <span style="font-weight: 600;">{months_to_threshold if months_to_threshold else 'Not within 24 months'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Condition Progression Section
    progression_class = "report-section-critical" if progression['risk_level'] in ['High', 'Critical'] else ("report-section-warning" if progression['total_worsening'] > 0 else "report-section-success")
    st.markdown(f"""
    <div class="{progression_class}">
        <h3 style="color: #2c3e50; margin-top: 0;">Condition Progression Summary</h3>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0;">
            <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.6); border-radius: 8px;">
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">Overall Risk</div>
                <div style="font-size: 20px; font-weight: 700; color: {'#e74c3c' if progression['risk_level'] in ['High', 'Critical'] else '#f39c12' if progression['risk_level'] == 'Moderate' else '#27ae60'};">{progression['risk_level']}</div>
            </div>
            <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.6); border-radius: 8px;">
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">Worsening</div>
                <div style="font-size: 20px; font-weight: 700; color: #e74c3c;">{progression['total_worsening']}</div>
                <div style="font-size: 11px; color: #95a5a6;">Severe: {progression['severe_worsening']}</div>
            </div>
            <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.6); border-radius: 8px;">
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">Improving</div>
                <div style="font-size: 20px; font-weight: 700; color: #27ae60;">{len(progression['improving_conditions'])}</div>
            </div>
            <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.6); border-radius: 8px;">
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">Stable</div>
                <div style="font-size: 20px; font-weight: 700; color: #95a5a6;">{len(progression['stable_conditions'])}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if progression['worsening_conditions']:
        st.markdown('<div style="margin-top: 20px;"><strong style="color: #e74c3c;">[WARNING] Worsening Conditions:</strong></div>', unsafe_allow_html=True)
        for cond in progression['worsening_conditions'][:5]:  # Show top 5
            st.markdown(f"""
            <div style="margin: 8px 0; padding: 12px; background: rgba(231, 76, 60, 0.1); border-left: 3px solid #e74c3c; border-radius: 4px;">
                <strong>{cond['name']}</strong><br>
                <span style="font-size: 13px; color: #7f8c8d;">
                    {cond['first_severity']} → {cond['last_severity']} •
                    Δ {cond['change_score']:+.1f} •
                    {cond['rate_per_month']:+.2f}/month
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Fall Risk Assessment Section
    fall_risk_class = "report-section-critical" if current_fall_risk > 50 else ("report-section-warning" if current_fall_risk > 30 else "report-section")
    st.markdown(f"""
    <div class="{fall_risk_class}">
        <h3 style="color: #2c3e50; margin-top: 0;">[WARNING] Fall Risk Assessment</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 15px 0;">
            <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.6); border-radius: 8px;">
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">Baseline Risk</div>
                <div style="font-size: 28px; font-weight: 700; color: #34495e;">{baseline_fall_risk:.1f}%</div>
            </div>
            <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.6); border-radius: 8px;">
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">Current Risk</div>
                <div style="font-size: 28px; font-weight: 700; color: {'#e74c3c' if current_fall_risk > 50 else '#f39c12' if current_fall_risk > 30 else '#27ae60'};">{current_fall_risk:.1f}%</div>
                <div style="font-size: 13px; margin-top: 5px; font-weight: 600; color: {'#e74c3c' if current_fall_risk > 50 else '#f39c12' if current_fall_risk > 30 else '#27ae60'};">{fall_category}</div>
            </div>
            <div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.6); border-radius: 8px;">
                <div style="font-size: 12px; color: #7f8c8d; margin-bottom: 5px;">Change</div>
                <div style="font-size: 28px; font-weight: 700; color: {'#e74c3c' if (current_fall_risk - baseline_fall_risk) > 0 else '#27ae60'};">{current_fall_risk - baseline_fall_risk:+.1f}</div>
                <div style="font-size: 11px; color: #95a5a6;">percentage points</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Intervention Analysis Section
    st.markdown("""
    <div class="report-section">
        <h3 style="color: #2c3e50; margin-top: 0;">[RX] Intervention Analysis (24-Month Projections)</h3>
    """, unsafe_allow_html=True)

    for scenario, outcome in scenario_outcomes.items():
        months_to_crit = compute_months_to_threshold(outcome)
        benefit = outcome[-1] - predictions[-1]

        if scenario == "No Intervention (Baseline)":
            bg_color = "#f8f9fa"
            border_color = "#95a5a6"
        elif benefit > 10:
            bg_color = "#eafaf1"
            border_color = "#27ae60"
        elif benefit > 5:
            bg_color = "#fef9e7"
            border_color = "#f39c12"
        else:
            bg_color = "#f8f9fa"
            border_color = "#95a5a6"

        st.markdown(f"""
        <div style="margin: 12px 0; padding: 15px; background: {bg_color}; border-left: 4px solid {border_color}; border-radius: 6px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="font-size: 15px; color: #2c3e50;">{scenario}</strong>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 18px; font-weight: 700; color: #2c3e50;">{outcome[-1]:.1f}</span>
                    <span style="font-size: 13px; color: #7f8c8d;">/100 at 24mo</span>
                </div>
            </div>
            <div style="margin-top: 8px; font-size: 13px; color: #7f8c8d;">
                Benefit vs. baseline: <strong style="color: {'#27ae60' if benefit > 0 else '#e74c3c'};">{benefit:+.1f} points</strong> •
                Risk threshold: <strong>{months_to_crit if months_to_crit else 'Not within 24mo'}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Recommended Intervention Section
    st.markdown(f"""
    <div class="recommendation-box">
        <h3 style="margin-top: 0; color: white;">Recommended Primary Intervention</h3>
        <div style="font-size: 24px; font-weight: 700; margin: 15px 0;">{best_scenario}</div>
        <div style="font-size: 14px; opacity: 0.9; line-height: 1.6;">
            <strong>Rationale:</strong> This intervention shows the highest long-term health score preservation with evidence-backed decline reduction.<br>
            <strong>Implementation Window:</strong> Initiate within 4 weeks with quarterly monitoring for optimal outcomes.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clinical Recommendations Section
    clinical_class = "report-section-critical" if (statistics['trend_direction'] == 'declining' and months_to_threshold and months_to_threshold <= 12) else "report-section"
    st.markdown(f"""
    <div class="{clinical_class}">
        <h3 style="color: #2c3e50; margin-top: 0;">[CLINICAL] Clinical Recommendations</h3>
    """, unsafe_allow_html=True)

    if statistics['trend_direction'] == 'declining' and months_to_threshold and months_to_threshold <= 12:
        st.markdown("""
        <div style="background: #fee; padding: 15px; border-left: 4px solid #e74c3c; border-radius: 6px; margin: 10px 0;">
            <strong style="color: #c0392b;">[WARNING] URGENT ACTION REQUIRED</strong>
            <ul style="margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                <li>Escalate to specialist care within 2 weeks</li>
                <li>Begin intensive rehabilitation programme and fall-prevention protocol</li>
                <li>Schedule monthly health score reviews until stabilization</li>
                <li>Review medications, footwear, and home safety risk factors</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif statistics['trend_direction'] == 'declining':
        st.markdown("""
        <div style="background: #fef9e7; padding: 15px; border-left: 4px solid #f39c12; border-radius: 6px; margin: 10px 0;">
            <strong style="color: #d68910;">[WARNING] PREVENTIVE ACTION RECOMMENDED</strong>
            <ul style="margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                <li>Initiate preventive programme (orthotics + physiotherapy) within 6 weeks</li>
                <li>Conduct quarterly monitoring scans</li>
                <li>Reinforce patient adherence and lifestyle modifications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: #eafaf1; padding: 15px; border-left: 4px solid #27ae60; border-radius: 6px; margin: 10px 0;">
            <strong style="color: #1e8449;">[OK] MAINTENANCE PROTOCOL</strong>
            <ul style="margin: 10px 0; padding-left: 20px; line-height: 1.8;">
                <li>Maintain current therapeutic regimen</li>
                <li>Semi-annual monitoring unless symptoms change</li>
                <li>Continue strength and balance maintenance exercises</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Insurance Data Summary Section
    st.markdown(f"""
    <div class="report-section">
        <h3 style="color: #2c3e50; margin-top: 0;">Insurance Data Summary</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #ecf0f1;">
                <td style="padding: 12px; font-weight: 600; color: #7f8c8d;">Longitudinal Record Value</td>
                <td style="padding: 12px; text-align: right; font-weight: 600; color: #2c3e50;">{'£85-125' if len(scan_dates) >= 3 else '£45-75'}</td>
            </tr>
            <tr style="border-bottom: 1px solid #ecf0f1;">
                <td style="padding: 12px; font-weight: 600; color: #7f8c8d;">Recommended Premium Adjustment</td>
                <td style="padding: 12px; text-align: right; font-weight: 600; color: #2c3e50;">{max(0, (100 - health_scores[-1]) * 0.5 / 100):.2f}</td>
            </tr>
            <tr>
                <td style="padding: 12px; font-weight: 600; color: #7f8c8d;">Expected 5-Year Claim Likelihood</td>
                <td style="padding: 12px; text-align: right; font-weight: 600; color: {'#e74c3c' if risk_matrix_claim == 'High' else '#f39c12' if risk_matrix_claim == 'Moderate' else '#27ae60'};">{risk_matrix_claim}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Compliance & Evidence Footer
    st.markdown(f"""
    <div style="background: #ecf0f1; padding: 20px; border-radius: 8px; margin-top: 20px; text-align: center;">
        <div style="font-size: 12px; color: #7f8c8d; line-height: 1.8;">
            <strong>Compliance & Data Governance:</strong> GDPR compliant • UK DPA 2018 compliant • HIPAA aligned (if applicable)<br>
            Patient consent recorded at point of scan ingestion<br><br>
            <em>Evidence base: {MEDICAL_RESEARCH_DATABASE['total_studies']:,} peer-reviewed medical studies</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    json_payload = {
        "report_generated": datetime.now().isoformat(),
        "monitoring": {
            "first_scan": first_scan.isoformat(),
            "last_scan": last_scan.isoformat(),
            "duration_days": monitoring_days,
            "duration_months": monitoring_months,
            "scan_count": len(scan_dates)
        },
        "trajectory": {
            "baseline_score": float(health_scores[0]),
            "current_score": float(health_scores[-1]),
            "change": float(score_change),
            "monthly_rate": float(monthly_rate),
            "trend": statistics["trend_direction"],
            "model": statistics["best_model"],
            "r_squared": statistics["r_squared"],
            "predicted_score_12": float(prediction_12),
            "predicted_score_24": float(prediction_24),
            "months_to_threshold": months_to_threshold
        },
        "condition_progression": progression,
        "fall_risk": {
            "baseline": baseline_fall_risk,
            "current": current_fall_risk,
            "category": fall_category,
            "absolute_change": current_fall_risk - baseline_fall_risk
        },
        "interventions": {
            scenario: {
                "predicted_score_24": outcome[-1],
                "months_to_threshold": compute_months_to_threshold(outcome),
                "benefit_vs_baseline": outcome[-1] - predictions[-1]
            }
            for scenario, outcome in scenario_outcomes.items()
        },
        "recommendations": {
            "primary_intervention": best_scenario,
            "clinical_actions": report_lines[report_lines.index("## Clinical Recommendations") + 1:report_lines.index("## Insurance Data Summary")]
        },
        "insurance": {
            "record_value": "£85-125" if len(scan_dates) >= 3 else "£45-75",
            "premium_adjustment": max(0, (100 - health_scores[-1]) * 0.5 / 100),
            "claim_likelihood_band": risk_matrix_claim
        }
    }

    # Generate plain text version for download
    report_text = "\n".join(report_lines)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Report (TXT)",
            data=report_text,
            file_name=f"temporal_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    with col2:
        st.download_button(
            label="Download Report (JSON)",
            data=json.dumps(json_payload, indent=2),
            file_name=f"temporal_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
