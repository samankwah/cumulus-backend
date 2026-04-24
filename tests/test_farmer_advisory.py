from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cumulus.advisory.farmer_rules import build_farmer_advisory
from cumulus.frontend_contract.serializers import serialize_farmer_advisory
from cumulus.settings import AdvisoryConfig


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "farmer_advisory"


def _load_case(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _build_frame(payload: dict[str, object]) -> pd.DataFrame:
    frame = pd.DataFrame(payload["daily_forecast"])
    frame = frame.rename(columns={"date": "time", "rainfall_mm": "rainfall_corrected_mm", "temperature_c": "temp_c"})
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame


def test_farmer_advisory_planting_transitions():
    config = AdvisoryConfig()

    plant_now_case = _load_case("plant_now")
    plant_now = build_farmer_advisory(_build_frame(plant_now_case["request"]), config)
    assert plant_now["planting_recommendation"]["level"] == "plant_now"

    delay_case = _load_case("delay_dry_spell")
    delay = build_farmer_advisory(_build_frame(delay_case["request"]), config)
    assert delay["planting_recommendation"]["level"] == "delay_due_to_dry_spell_risk"

    wait_case = _load_case("irrigation_hot")
    wait = build_farmer_advisory(_build_frame(wait_case["request"]), config)
    assert wait["planting_recommendation"]["level"] == "wait_for_more_rain"


def test_farmer_advisory_dry_spell_severity_bands():
    config = AdvisoryConfig()

    no_alert_case = _load_case("plant_now")
    no_alert = build_farmer_advisory(_build_frame(no_alert_case["request"]), config)
    assert no_alert["dry_spell_alert"]["level"] == "none"

    watch_case = _load_case("irrigation_hot")
    watch = build_farmer_advisory(_build_frame(watch_case["request"]), config)
    assert watch["dry_spell_alert"]["level"] == "watch"
    assert watch["dry_spell_alert"]["dry_spell_length_days"] == 6

    warning_case = _load_case("delay_dry_spell")
    warning = build_farmer_advisory(_build_frame(warning_case["request"]), config)
    assert warning["dry_spell_alert"]["level"] == "warning"
    assert warning["dry_spell_alert"]["dry_spell_length_days"] == 7


def test_farmer_advisory_irrigation_escalates_with_hot_conditions():
    config = AdvisoryConfig()

    hot_case = _load_case("irrigation_hot")
    hot = build_farmer_advisory(_build_frame(hot_case["request"]), config)
    assert hot["irrigation_advice"]["level"] == "irrigate_if_possible"
    assert hot["irrigation_advice"]["temperature_band"] == "high_stress"


def test_farmer_advisory_content_is_short_and_non_empty():
    payload = build_farmer_advisory(_build_frame(_load_case("plant_now")["request"]), AdvisoryConfig())
    for key in ("planting_recommendation", "dry_spell_alert", "irrigation_advice"):
        block = payload[key]
        assert block["headline"]
        assert block["recommendation"]
        assert block["reason"]
        assert len(block["headline"]) <= 60
        assert len(block["recommendation"]) <= 100


def test_farmer_advisory_examples_match_documented_outputs():
    config = AdvisoryConfig()
    for name in ("plant_now", "delay_dry_spell", "irrigation_hot"):
        case = _load_case(name)
        response = build_farmer_advisory(_build_frame(case["request"]), config)
        response["location_id"] = case["request"]["location_id"]
        assert serialize_farmer_advisory(response).model_dump() == case["response"]
