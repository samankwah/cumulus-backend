"""Reusable evaluation framework for weather-agriculture advisory systems."""

from __future__ import annotations

from typing import Any

DEFAULT_REPORTING_CUTS = [
    "district",
    "region",
    "lead_time",
    "season",
    "advisory_type",
    "farmer_segment",
]

DEFAULT_EVALUATION_MODES = [
    {
        "name": "offline_backtesting",
        "purpose": "Evaluate forecast and advisory logic on historical weather and observed outcomes before deployment.",
    },
    {
        "name": "prospective_pilot",
        "purpose": "Issue live advisories during a pilot and compare later farm outcomes and farmer responses.",
    },
    {
        "name": "continuous_monitoring",
        "purpose": "Track monthly operations and seasonal impact after rollout.",
    },
]

DEFAULT_FEEDBACK_QUESTIONS = [
    "Did you receive the advisory?",
    "Did you understand it?",
    "Did you follow it?",
    "If not, why not?",
    "Was it useful?",
    "What happened after you acted?",
]

DEFAULT_NON_ADOPTION_REASONS = [
    "lack_of_inputs",
    "labor_constraints",
    "distrust",
    "conflicting_local_knowledge",
    "message_arrived_too_late",
    "message_unclear",
]

DEFAULT_SCENARIOS = [
    "High numerical forecast accuracy but low farmer usefulness",
    "Low numerical forecast accuracy but acceptable decision usefulness",
    "Good rainfall MAE but poor planting outcomes",
    "Strong advisory quality in one region but weak in another",
    "Low adoption despite strong outcomes among adopters",
    "Delay planting recommendations that reduce failed establishment",
]


def build_evaluation_framework(
    *,
    reference_system: str = "Ghana maize advisory product",
    reference_crop: str = "maize",
    reference_country: str = "Ghana",
) -> dict[str, Any]:
    """Return a reusable framework specification anchored to the current product."""

    return {
        "reference_implementation": {
            "system": reference_system,
            "country": reference_country,
            "crop": reference_crop,
            "design_note": (
                "The framework is reusable across crops, regions, and advisory products, "
                "but the current Ghana maize system is the default reference implementation."
            ),
        },
        "layers": [
            {
                "name": "forecast_performance",
                "question": "Was the rainfall forecast numerically and operationally correct?",
                "core_metrics": [
                    {"name": "MAE (mm)", "interpretation": "Mean absolute rainfall error by day and horizon."},
                    {"name": "RMSE (mm)", "interpretation": "Penalizes larger misses and heavy-rain errors."},
                    {"name": "Bias (mm)", "interpretation": "Detects systematic underprediction or overprediction."},
                    {"name": "Correlation", "interpretation": "Checks whether predicted and observed rainfall move together."},
                    {
                        "name": "Wet-day detection precision/recall/F1",
                        "interpretation": "Evaluates rain/no-rain detection using a configurable wet-day threshold.",
                    },
                    {
                        "name": "Heavy-rain event recall",
                        "interpretation": "Measures recall on operationally important rainfall events.",
                    },
                    {
                        "name": "Dry-spell detection accuracy",
                        "interpretation": "Compares predicted versus observed longest dry runs inside the guard window.",
                    },
                    {
                        "name": "Calibration by rainfall band",
                        "interpretation": "Compares predicted and observed low/medium/high rainfall frequencies.",
                    },
                ],
            },
            {
                "name": "advisory_decision_performance",
                "question": "Did forecast-driven decisions recommend the right action at the right time?",
                "core_metrics": [
                    {
                        "name": "Planting success rate",
                        "interpretation": "Share of advised-and-executed planting actions that establish successfully.",
                    },
                    {
                        "name": "Recommended-planting success rate",
                        "interpretation": "Success rate when the advisory said plant and the farmer planted.",
                    },
                    {
                        "name": "Delayed-planting avoided-failure rate",
                        "interpretation": "Share of delay recommendations that prevented a poor planting window.",
                    },
                    {
                        "name": "False-go planting rate",
                        "interpretation": "Share of plant-now recommendations that still led to failed establishment.",
                    },
                    {
                        "name": "Missed-opportunity rate",
                        "interpretation": "Share of delay recommendations issued when the window would have worked.",
                    },
                    {
                        "name": "Net planting uplift",
                        "interpretation": "Difference in planting success for advisory-timed planting versus baseline practice.",
                    },
                ],
                "advisory_families": [
                    "planting_recommendation",
                    "dry_spell_alert",
                    "irrigation_advice",
                ],
            },
            {
                "name": "farmer_usefulness_and_adoption",
                "question": "Did farmers understand, trust, act on, and benefit from the advice?",
                "core_metrics": [
                    {
                        "name": "Advisory actionability rate",
                        "interpretation": "Share of advisories farmers say were clear enough to act on.",
                    },
                    {"name": "Adoption rate", "interpretation": "Share of advisories followed fully or partially."},
                    {
                        "name": "Perceived usefulness score",
                        "interpretation": "Structured 1-5 rating on usefulness, clarity, and trust.",
                    },
                    {
                        "name": "Decision concordance",
                        "interpretation": "Share of cases where farmer action matched advisory intent.",
                    },
                    {
                        "name": "Outcome lift",
                        "interpretation": "Difference in success or avoided losses for users versus comparison groups.",
                    },
                ],
            },
        ],
        "evaluation_modes": list(DEFAULT_EVALUATION_MODES),
        "required_reporting_cuts": list(DEFAULT_REPORTING_CUTS),
        "data_streams": [
            {
                "name": "system_and_weather_truth",
                "description": "Store every forecast and advisory exactly as issued and join it to post-event weather truth.",
                "minimum_tables": [
                    {
                        "table": "forecast_issuance_log",
                        "fields": [
                            "forecast_id",
                            "issued_at",
                            "location_id",
                            "district",
                            "region",
                            "agro_ecological_zone",
                            "lead_time",
                            "target_date",
                            "model_version",
                            "predicted_rainfall_mm",
                        ],
                    },
                    {
                        "table": "advisory_issuance_log",
                        "fields": [
                            "advisory_id",
                            "forecast_id",
                            "issued_at",
                            "location_id",
                            "advisory_type",
                            "recommendation_level",
                            "recommended_action",
                            "message_channel",
                            "message_version",
                        ],
                    },
                    {
                        "table": "observed_weather_daily",
                        "fields": [
                            "location_id",
                            "date",
                            "observed_rainfall_mm",
                            "truth_source",
                        ],
                    },
                ],
            },
            {
                "name": "farm_outcomes",
                "description": "Collect the minimum field outcomes needed to evaluate planting advice.",
                "minimum_tables": [
                    {
                        "table": "farm_outcomes",
                        "fields": [
                            "farmer_id",
                            "plot_id",
                            "location_id",
                            "crop_type",
                            "variety",
                            "planting_date",
                            "establishment_success",
                            "replanting_indicator",
                            "failure_reason",
                            "yield_or_yield_class",
                        ],
                    }
                ],
            },
            {
                "name": "farmer_feedback",
                "description": "Collect structured and qualitative response data through multiple channels.",
                "minimum_tables": [
                    {
                        "table": "farmer_feedback",
                        "fields": [
                            "feedback_id",
                            "advisory_id",
                            "farmer_id",
                            "response_time",
                            "received_advisory",
                            "understood_advisory",
                            "followed_advice",
                            "non_adoption_reason",
                            "usefulness_score",
                            "clarity_score",
                            "trust_score",
                            "free_text_feedback",
                        ],
                    }
                ],
                "channels": [
                    "sms_ussd_follow_up",
                    "enumerator_phone_surveys",
                    "extension_agent_check_ins",
                    "app_or_whatsapp_feedback",
                    "end_of_season_focus_groups",
                ],
                "recommended_questions": list(DEFAULT_FEEDBACK_QUESTIONS),
                "structured_non_adoption_reasons": list(DEFAULT_NON_ADOPTION_REASONS),
                "feedback_cadence": [
                    "immediate_post_message_within_24_to_72_hours",
                    "event_follow_up_after_planting_or_dry_spell",
                    "seasonal_retrospective_after_harvest_or_establishment_window",
                ],
            },
        ],
        "review_cadence": {
            "monthly_operational_review": [
                "forecast performance by lead time, district, region, and season",
                "advisory issuance volume and coverage",
                "message delivery delays and failure rates",
                "farmer actionability, adoption, and trust indicators",
            ],
            "seasonal_impact_review": [
                "planting success and avoided replanting outcomes",
                "outcome lift versus baseline or comparison cohort",
                "performance by agro_ecological_zone and farmer segment",
                "threshold recalibration and advisory-rule updates",
            ],
        },
        "supported_scenarios": list(DEFAULT_SCENARIOS),
        "assumptions_and_defaults": [
            "The default reference system is the Ghana maize advisory product, but thresholds should remain swappable.",
            "The default planting outcome is establishment success or failure rather than yield.",
            "The default impact baseline is current farmer practice or a non-advisory comparison cohort.",
            "If farm outcomes are sparse at launch, start with forecast metrics plus adoption and usefulness metrics.",
            "Farmer feedback collection should be multimodal because no single channel will cover all users reliably.",
        ],
        "acceptance_criteria": [
            "Define mandatory core metrics for rainfall and planting success.",
            "Distinguish forecast accuracy from advisory usefulness.",
            "Specify minimum required data tables and logs for offline and field evaluation.",
            "Include a farmer feedback loop with structured and qualitative channels.",
            "Support monthly operational review and seasonal impact review.",
        ],
    }


def render_evaluation_framework_markdown(framework: dict[str, Any]) -> str:
    """Render the framework as a Markdown brief for humans."""

    reference = framework["reference_implementation"]
    lines = [
        "# Evaluation Framework For A Weather-Based Agricultural Advisory System",
        "",
        "## Summary",
        (
            "This framework is reusable across weather-agriculture advisory products, "
            f"with the current {reference['country']} {reference['crop']} system as the reference implementation."
        ),
        "",
        "## Three-Layer Scorecard",
    ]
    for layer in framework["layers"]:
        lines.append(f"### {layer['name']}")
        lines.append(layer["question"])
        lines.append("")
        for metric in layer["core_metrics"]:
            lines.append(f"- **{metric['name']}**: {metric['interpretation']}")
        advisory_families = layer.get("advisory_families")
        if advisory_families:
            lines.append("")
            lines.append(f"Advisory families: {', '.join(advisory_families)}")
        lines.append("")

    lines.extend(
        [
            "## Evaluation Modes",
            "",
        ]
    )
    for mode in framework["evaluation_modes"]:
        lines.append(f"- **{mode['name']}**: {mode['purpose']}")

    lines.extend(
        [
            "",
            "## Required Reporting Cuts",
            "",
            f"- {', '.join(framework['required_reporting_cuts'])}",
            "",
            "## Minimum Required Data Streams",
            "",
        ]
    )
    for stream in framework["data_streams"]:
        lines.append(f"### {stream['name']}")
        lines.append(stream["description"])
        lines.append("")
        for table in stream["minimum_tables"]:
            lines.append(f"- **{table['table']}**: {', '.join(table['fields'])}")
        for key in ("channels", "recommended_questions", "structured_non_adoption_reasons", "feedback_cadence"):
            values = stream.get(key)
            if values:
                lines.append(f"- **{key}**: {', '.join(values)}")
        lines.append("")

    lines.extend(
        [
            "## Review Cadence",
            "",
            "### Monthly Operational Review",
        ]
    )
    for item in framework["review_cadence"]["monthly_operational_review"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "### Seasonal Impact Review",
        ]
    )
    for item in framework["review_cadence"]["seasonal_impact_review"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Supported Scenarios",
            "",
        ]
    )
    for scenario in framework["supported_scenarios"]:
        lines.append(f"- {scenario}")

    lines.extend(
        [
            "",
            "## Assumptions And Defaults",
            "",
        ]
    )
    for assumption in framework["assumptions_and_defaults"]:
        lines.append(f"- {assumption}")

    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "",
        ]
    )
    for item in framework["acceptance_criteria"]:
        lines.append(f"- {item}")

    return "\n".join(lines)
