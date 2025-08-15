from ai_vacation_companion.core.recommender import Preference, normalize_tags, recommend
from ai_vacation_companion.core.data_loader import load_json

def test_basic_recommendation():
    dests = load_json("destinations.json")
    pkgs = load_json("packages.json")
    pref = Preference(budget="medium", climate="warm", activities=["beach", "culture"], duration_days=6)
    res = recommend(pref, dests, pkgs, top_k=3)
    assert len(res) >= 1
