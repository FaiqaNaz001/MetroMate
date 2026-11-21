"""
test_chat_tools.py
------------------
Checks that the chatbot's tools follow the rules we agreed. Run with:
    python test_chat_tools.py

These tests check the FACTS and the OFFERS the code hands to the AI. They can't check how
the AI words its reply; that is what the hand-run question list is for.
"""

import json

import chat_tools
from chat_tools import (
    plan_trip, get_bus_details, get_metro_taxi_option, places_near_station,
    plan_route, park_and_ride, run_tool, compact_route, TOOLS, TOOL_FUNCTIONS,
)
from landmarks import get_landmark_info


def as_text(result):
    """The result exactly as the AI would see it."""
    return json.dumps(result, separators=(",", ":"))


# ---------- the rules for each situation ----------

def test_walkable_place_walks_and_has_a_taxi_distance():
    r = plan_trip("Katara", "Hyatt Plaza")              # 1000 m walk, 1300 m by road, has bus M313
    assert r["last_stretch"]["advice"] == "walk"
    assert r["last_stretch"]["walkable"] is True
    assert r["last_stretch"]["road_m"] == 1300           # for riders who'd rather not walk
    assert r["last_stretch"]["walk_min"] == 12


def test_very_short_walk_has_no_taxi_distance_and_no_offer():
    r = plan_trip("Katara", "Mall of Qatar")            # 300 m: nobody takes a taxi
    assert r["last_stretch"]["advice"] == "walk"
    assert "road_m" not in r["last_stretch"]
    assert "offer" not in r


def test_walkable_place_with_a_bus_still_offers_it():
    r = plan_trip("Katara", "Hyatt Plaza")
    assert r["offer"]["type"] == "bus_same_station"
    assert "free bus service" in r["offer"]["question"]
    assert "Al Aziziyah" in r["offer"]["question"]


def test_far_place_gets_a_taxi_from_the_closest_station():
    r = plan_trip("Katara", "Doha Festival City")        # 7700 m walk
    assert r["last_stretch"]["advice"] == "taxi"
    assert r["last_stretch"]["walkable"] is False
    assert r["last_stretch"]["road_m"] == 8100
    assert r["offer"]["type"] == "bus_same_station"


def test_bus_from_a_different_station_is_offered_without_naming_it():
    r = plan_trip("Katara", "Aberdeen")                  # closest Bin Mahmoud, bus boards at Msheireb
    assert r["last_stretch"]["advice"] == "taxi"         # taxi from the closest station comes first
    assert r["closest_station"] == "Bin Mahmoud"
    assert r["offer"]["type"] == "bus_other_station"
    assert "different station" in r["offer"]["question"]
    assert "Msheireb" not in r["offer"]["question"]      # details stay hidden


def test_far_place_without_a_bus_has_no_offer():
    r = plan_trip("Katara", "Asmakh Mall")
    assert r["last_stretch"]["advice"] == "taxi"
    assert "offer" not in r


def test_tram_place_always_has_the_notice_and_a_taxi_offer():
    r = plan_trip("Al Sadd", "Place Vendome")            # tram, no bus in the data
    assert r["closest_station"] == "Tarfat South"
    assert r["last_stretch"]["station_type"] == "tram"
    assert "Legtaifiya" in r["tram_notice"]
    assert r["offer"]["type"] == "tram_taxi"
    assert "Lusail QNB" in r["offer"]["question"]
    assert "bus" not in r["offer"]["question"].lower()   # no bus exists for this place


def test_tram_place_with_a_bus_offers_both_in_one_question():
    r = plan_trip("Al Sadd", "Lusail University")        # tram + bus M147 from Qatar University
    assert "tram_notice" in r
    assert r["offer"]["type"] == "tram_alternatives"
    question = r["offer"]["question"].lower()
    assert "taxi" in question and "bus" in question
    assert "M147" not in as_text(r)                      # bus details stay hidden


def test_tram_time_is_unknown_but_metro_time_is_given():
    route = plan_trip("Al Sadd", "Place Vendome")["route"]
    assert route["tram_time_unknown"] is True
    assert route["metro_minutes"] > 0
    assert "minutes" not in route


def test_metro_only_trip_gives_total_minutes():
    route = plan_trip("Katara", "Mall of Qatar")["route"]
    assert route["minutes"] > 0
    assert "tram_time_unknown" not in route


def test_exactly_one_offer_per_reply():
    # The result has a single "offer" entry, never a list.
    for place in ("Hyatt Plaza", "Aberdeen", "Lusail University", "Place Vendome", "UDST"):
        offer = plan_trip("Katara", place).get("offer")
        assert isinstance(offer, dict) and set(offer) == {"type", "question"}


# ---------- hidden details ----------

def test_plan_trip_hides_bus_and_metro_taxi_details():
    aberdeen = as_text(plan_trip("Katara", "Aberdeen"))
    assert "M103" not in aberdeen and "get_off" not in aberdeen
    hyatt = as_text(plan_trip("Katara", "Hyatt Plaza"))
    assert "M313" not in hyatt
    fallback = get_landmark_info("MALL05")["fallback"]
    vendome = as_text(plan_trip("Al Sadd", "Place Vendome"))
    assert str(fallback["road_dist"]) not in vendome     # the metro-taxi distance is hidden


def test_bus_details_from_a_different_station():
    r = get_bus_details("Katara", "Aberdeen")
    bus = r["buses"][0]
    assert bus["bus"].startswith("M103") and "free" in bus["bus"]
    assert bus["board_at"] == "Msheireb station"
    assert bus["same_station_as_closest"] is False
    assert bus["get_off_at_stop"]
    assert bus["route_to_boarding_station"]["steps"]     # its own route to the boarding station
    assert "tap_in_note" in r
    assert "no bus timing" in r["timing"].lower()


def test_bus_details_from_the_same_station_have_no_extra_route():
    bus = get_bus_details("Katara", "UDST")["buses"][0]
    assert bus["same_station_as_closest"] is True
    assert "route_to_boarding_station" not in bus


def test_bus_details_work_for_walkable_places_too():
    assert get_bus_details("Katara", "Hyatt Plaza")["buses"][0]["bus"].startswith("M313")


def test_bus_details_for_a_place_without_a_bus_say_so():
    assert get_bus_details("Katara", "Asmakh Mall")["error"] == "no_bus_info"


def test_metro_taxi_option_for_a_tram_place():
    r = get_metro_taxi_option("Katara", "Place Vendome")
    assert r["metro_station"] == "Lusail QNB"
    assert r["advice"] == "taxi" and r["road_m"] > 0
    assert all("tram" not in step.lower() for step in r["route"]["steps"])   # metro only


def test_metro_taxi_option_for_other_places_is_an_error():
    assert get_metro_taxi_option("Katara", "Mall of Qatar")["error"] == "no_metro_option"


# ---------- Qatar University ----------

def test_qu_campus_asks_for_a_building_and_lists_them():
    r = plan_trip("Katara", "QU")
    assert r["needs_building"] is True
    assert len(r["buildings"]) == 15 and r["building_count"] == 15
    assert "College of Engineering" in r["buildings"]
    assert "last_stretch" not in r and "offer" not in r


def test_qu_building_gets_taxi_facts_and_no_offer():
    r = plan_trip("Katara", "QU Engineering")
    assert r["last_stretch"]["advice"] == "taxi" and "road_m" in r["last_stretch"]
    assert "offer" not in r and "needs_building" not in r


def test_qu_shuttle_link_appears_for_qu_places_only():
    assert chat_tools.QU_SHUTTLE_URL.startswith("https://")
    for place in ("QU", "QU Engineering", "QU Library", "Male Housing"):
        assert plan_trip("Katara", place)["qu_shuttle"]["url"] == chat_tools.QU_SHUTTLE_URL
    for place in ("Villaggio", "UDST", "Aberdeen"):
        assert "qu_shuttle" not in plan_trip("Katara", place)


def test_qu_shuttle_is_left_out_when_no_link_is_set():
    saved = chat_tools.QU_SHUTTLE_URL
    chat_tools.QU_SHUTTLE_URL = ""
    try:
        assert "qu_shuttle" not in plan_trip("Katara", "QU")
    finally:
        chat_tools.QU_SHUTTLE_URL = saved


def test_unlisted_qu_building_gets_the_route_and_the_list_of_real_buildings():
    r = plan_trip("Katara", "College of Pharmacy at QU")     # not in the data
    assert r["requested_building_not_listed"] == "College of Pharmacy at QU"
    assert r["needs_building"] is True and r["building_count"] == 15
    assert r["route"]["steps"] and "last_stretch" not in r and "offer" not in r
    assert not any("pharmacy" in name.lower() for name in r["buildings"])   # never invented
    assert r["qu_shuttle"]["url"] == chat_tools.QU_SHUTTLE_URL


def test_qu_building_phrasings_that_used_to_fail_now_work():
    for phrase in ("the research complex at QU", "QU research complex", "the library at QU"):
        r = plan_trip("Katara", phrase)
        assert "needs_building" not in r and r["last_stretch"]["advice"] == "taxi", phrase


def test_an_unknown_place_without_qu_gets_the_generic_hint():
    r = plan_trip("Katara", "College of Pharmacy")           # no mention of QU
    assert r["error"] == "place_not_found"
    assert "Qatar University" in r["hint"]


# ---------- errors and the other tools ----------

def test_plan_trip_errors():
    assert plan_trip("Narnia", "Mall of Qatar")["error"] == "start_not_found"
    assert plan_trip("Katara", "Narnia")["error"] == "place_not_found"
    r = plan_trip("Katara", "mall")
    assert r["error"] == "ambiguous_place" and len(r["options"]) >= 2


def test_places_near_station():
    r = places_near_station("Qatar National Library")
    names = [p["name"] for p in r["places"]]
    assert "Georgetown University in Qatar" in names
    assert [p["walk_m"] for p in r["places"]] == sorted(p["walk_m"] for p in r["places"])
    big = places_near_station("Qatar University")        # 15 buildings: list is capped
    assert len(big["places"]) == chat_tools.MAX_NEAR and big["more"] == big["count"] - chat_tools.MAX_NEAR
    assert places_near_station("Ras Bu Abboud")["places"] == []
    assert places_near_station("Narnia")["error"] == "station_not_found"


def test_plan_route_between_stations():
    r = plan_route("Katara", "Al Riffa")
    assert len(r["route"]["steps"]) == 2 and "tram_notice" not in r
    assert "tram_notice" in plan_route("Wadi", "Katara")
    assert plan_route("Narnia", "Katara")["error"] == "start_not_found"
    assert plan_route("Katara", "Narnia")["error"] == "end_not_found"


def test_starting_at_the_closest_station():
    r = plan_trip("Al Riffa", "Mall of Qatar")
    assert r["route"] == {"already_there": True}
    assert r["last_stretch"]["advice"] == "walk"


# ---------- Park & Ride ----------

def with_park_and_ride(data, check):
    """Run `check` with made-up Park & Ride data, then restore the real data."""
    saved = dict(chat_tools.PARK_AND_RIDE)
    chat_tools.PARK_AND_RIDE.clear()
    chat_tools.PARK_AND_RIDE.update(data)
    try:
        check()
    finally:
        chat_tools.PARK_AND_RIDE.clear()
        chat_tools.PARK_AND_RIDE.update(saved)


def test_park_and_ride_for_one_station():
    def check():
        assert park_and_ride("Katara") == {"station": "Katara", "park_and_ride": True}
        assert park_and_ride("al sadd")["park_and_ride"] is False
        blank = park_and_ride("Msheireb")                # filled in for others, blank for this one
        assert blank["park_and_ride"] == "unknown" and "note" in blank
        assert park_and_ride("Narnia")["error"] == "station_not_found"
    with_park_and_ride({"M_KATARA": True, "M_AL_SADD": False}, check)


def test_park_and_ride_list_and_tram_stations():
    def check():
        for word in ("all", "ALL", ""):
            assert park_and_ride(word) == {"stations_with_park_and_ride": ["Katara", "Lusail QNB"]}
        tram = park_and_ride("Wadi")                      # tram stations have no value
        assert tram["park_and_ride"] == "unknown" and "metro" in tram["note"]
    with_park_and_ride({"M_KATARA": True, "M_LUSAIL_QNB": True, "M_AL_SADD": False}, check)


def test_park_and_ride_without_any_data_says_so():
    def check():
        assert park_and_ride("Katara")["error"] == "no_park_and_ride_data"
        assert park_and_ride("all")["error"] == "no_park_and_ride_data"
    with_park_and_ride({}, check)


def test_park_and_ride_works_through_run_tool():
    def check():
        result = json.loads(run_tool("park_and_ride", '{"station": "Katara"}'))
        assert result["park_and_ride"] is True
    with_park_and_ride({"M_KATARA": True}, check)


# ---------- running tools the way the AI does ----------

def test_run_tool_handles_good_and_bad_calls():
    good = json.loads(run_tool("plan_trip", '{"start": "Katara", "place": "Mall of Qatar"}'))
    assert good["place"] == "Mall of Qatar"
    assert json.loads(run_tool("nope", "{}"))["error"] == "unknown_tool"
    assert json.loads(run_tool("plan_trip", "not json"))["error"] == "bad_arguments"
    assert json.loads(run_tool("plan_trip", '{"start": "Katara"}'))["error"] == "bad_arguments"
    assert json.loads(run_tool("plan_trip", '{"start": "Katara", "place": "x", "extra": 1}'))["error"] == "bad_arguments"


def test_tool_descriptions_match_the_functions():
    assert {t["function"]["name"] for t in TOOLS} == set(TOOL_FUNCTIONS)
    for tool in TOOLS:
        function = TOOL_FUNCTIONS[tool["function"]["name"]]
        assert set(tool["function"]["parameters"]["required"]) == set(function.__code__.co_varnames[:function.__code__.co_argcount])


# ---------- keeping results small (the free plan has a tokens-per-minute limit) ----------

def test_results_stay_small():
    worst = [plan_trip("Al Sadd", "Lusail University"), plan_trip("Katara", "QU"),
             get_bus_details("Katara", "Aberdeen"), get_metro_taxi_option("Al Sadd", "Place Vendome")]
    for result in worst:
        assert len(as_text(result)) < 1500, as_text(result)[:80]
    assert len(json.dumps(TOOLS, separators=(",", ":"))) < 3500


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as error:
            failed += 1
            print(f"FAIL  {name}  {error}")
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed.")
