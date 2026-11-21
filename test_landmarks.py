"""
test_landmarks.py
-----------------
Checks for landmarks.py. Run with:   python test_landmarks.py

Three kinds of tests:
  * Behaviour tests - does the lookup, the walk/taxi/bus rules and the tram notice work?
  * Bus tests - are buses shown only when they should be, and routed from the right station?
  * Data tests - is the CSV data consistent? (These catch typos in your files.)
"""

from landmarks import (
    LANDMARKS, LANDMARK_STATIONS, BUSES, STATIONS, PARK_AND_RIDE, WALK_LIMIT_M,
    find_landmark, transport_suggestion, get_landmark_info,
    plan_trip_to_landmark, parse_meters, parse_flag,
)


class Skip(Exception):
    """Raised by a test that can't run yet. The runner reports it as SKIP, not PASS."""


def lookup(text):
    return find_landmark(text, LANDMARKS)


# ---------- finding places ----------

def test_finds_by_name_alias_and_any_capitalisation():
    assert lookup("Mall of Qatar") == ("found", ["MALL01"])
    assert lookup("moq") == ("found", ["MALL01"])
    assert lookup("  VILLAGGIO  ") == ("found", ["MALL02"])
    assert lookup("HBKU") == ("found", ["UNI08"])
    assert lookup("City Center") == ("found", ["MALL15"])


def test_qatar_university_and_its_buildings():
    assert lookup("QU") == ("found", ["UNI17"])
    assert lookup("Qatar University") == ("found", ["UNI17"])
    assert lookup("QU Engineering") == ("found", ["QU_H07"])
    assert lookup("QU library") == ("found", ["QU_B13"])
    assert lookup("QU Male Dorm") == ("found", ["QU_F05"])
    assert lookup("QU Sports Complex") == ("found", ["QU_A07"])


def test_longer_and_reworded_phrasings_still_match():
    # Things people really type, which the old letter-matching missed.
    assert lookup("the research complex") == ("found", ["QU_H10"])
    assert lookup("research complex at QU") == ("found", ["QU_H10"])
    assert lookup("the library at QU") == ("found", ["QU_B13"])
    assert lookup("Sports and Events Complex") == ("found", ["QU_A07"])   # "and" vs "&"
    assert lookup("law college") == ("found", ["QU_I09"])                  # word order
    assert lookup("business college") == ("found", ["QU_H08"])
    assert lookup("engineering building") == ("found", ["QU_H07"])
    assert lookup("H10") == ("found", ["QU_H10"])
    assert lookup("the mall") == ("found", ["MALL13"])                     # the mall called "The Mall"


def test_matching_is_by_whole_words_not_letters():
    assert lookup("male dorm") == ("found", ["QU_F05"])      # not also "fe-male" dorm
    assert lookup("female dorm") == ("found", ["QU_F16"])


def test_places_that_are_not_in_the_data_are_not_matched():
    for name in ("College of Pharmacy", "pharmacy at QU", "Narnia", "al"):
        assert lookup(name)[0] == "not_found", name


def test_bare_category_words_lead_to_which_one():
    for word in ("mall", "university", "college"):
        assert lookup(word)[0] == "ambiguous", word


def test_unknown_and_vague_names():
    assert lookup("Narnia")[0] == "not_found"
    assert lookup("")[0] == "not_found"
    assert lookup("mall")[0] == "ambiguous"        # many malls match
    assert lookup("university")[0] == "ambiguous"  # many universities match


# ---------- the walk / taxi rule ----------

def test_walk_limit_rule():
    assert transport_suggestion(300) == "walk"
    assert transport_suggestion(WALK_LIMIT_M) == "walk"       # exactly the limit still walks
    assert transport_suggestion(WALK_LIMIT_M + 1) == "taxi"
    assert transport_suggestion(7700) == "taxi"


def test_landmark_info_uses_the_rule():
    assert get_landmark_info("MALL01")["suggestion"] == "walk"    # Mall of Qatar, 300 m
    assert get_landmark_info("MALL07")["suggestion"] == "walk"    # Hyatt Plaza, exactly 1000 m
    assert get_landmark_info("MALL08")["suggestion"] == "taxi"    # Doha Festival City, 7700 m
    assert get_landmark_info("QU_H07")["suggestion"] == "taxi"    # QU Engineering, 1300 m


def test_both_distances_are_available_even_when_far():
    info = get_landmark_info("MALL08")
    assert info["walk_dist"] is not None and info["road_dist"] is not None


def test_very_short_walks_have_no_road_distance():
    assert get_landmark_info("MALL01")["road_dist"] is None   # 300 m: nobody drives
    assert get_landmark_info("MALL02")["road_dist"] is None   # 100 m


def test_qatar_university_campus_asks_for_a_building():
    info = get_landmark_info("UNI17")
    assert info["status"] == "needs_detail"
    assert len(info["options"]) == 15
    assert "College of Engineering" in info["options"]
    assert "Male Housing" in info["options"]
    assert info["station_name"] == "Qatar University"


# ---------- tram places and their metro fallback ----------

def test_tram_landmarks_have_a_tram_station_and_a_metro_fallback():
    for lid, tram_station in [("MALL05", "T_TARFAT_SOUTH"),
                              ("UNI02", "T_AL_WESSIL"),
                              ("UNI15", "T_AL_WESSIL")]:
        info = get_landmark_info(lid)
        assert info["station_id"] == tram_station
        assert info["fallback"]["station_id"].startswith("M_")   # fallback is a metro station
        assert info["fallback"]["suggestion"] == "taxi"          # these are all long walks


def test_places_without_a_tram_have_no_fallback():
    assert "fallback" not in get_landmark_info("MALL01")
    assert "fallback" not in get_landmark_info("UNI17")


# ---------- buses (always an offer, listed for every place that has one) ----------

def test_bus_from_the_same_station():
    info = get_landmark_info("UNI01")                    # UDST: M148 boards at Qatar University
    assert info["buses"][0]["same_station"] is True


def test_bus_from_a_different_station():
    info = get_landmark_info("UNI12")                    # Aberdeen: closest is Bin Mahmoud, M103 boards at Msheireb
    assert info["station_id"] == "M_BIN_MAHMOUD"
    assert info["buses"][0]["from_station_id"] == "M_MSHEIREB"
    assert info["buses"][0]["same_station"] is False
    assert info["suggestion"] == "taxi"                  # taxi from the closest station still comes first


def test_far_place_without_a_bus_has_an_empty_bus_list():
    assert get_landmark_info("MALL11")["buses"] == []    # Asmakh Mall, 1300 m, no bus


def test_walkable_place_still_lists_its_bus():
    info = get_landmark_info("MALL07")                   # Hyatt Plaza: walkable, but has bus M313
    assert info["suggestion"] == "walk"
    assert len(info["buses"]) == 1


def test_tram_places_list_the_bus_backup():
    for lid in ("UNI02", "UNI15"):
        info = get_landmark_info(lid)
        assert info["buses"][0]["bus_number"] == "M147"
        assert info["buses"][0]["from_station_id"] == "M_QATAR_UNIVERSITY"
        assert info["buses"][0]["same_station"] is False


def test_metrolink_buses_are_flagged():
    for buses in BUSES.values():
        for bus in buses:
            assert bus["bus_number"].upper().startswith("M")   # all current buses are Metrolink
    assert get_landmark_info("UNI01")["buses"][0]["is_metrolink"] is True


# ---------- planning whole trips ----------

def test_trip_to_mall_of_qatar_has_no_tram_notice_or_bus():
    r = plan_trip_to_landmark("Katara", "Mall of Qatar")
    assert r["status"] == "ok"
    assert [s["line"] for s in r["route"]["segments"]] == ["RED", "GREEN"]
    assert r["landmark"]["suggestion"] == "walk"
    assert r["uses_tram"] is False
    assert r["tram_notice"] is None
    assert r["fallback_route"] is None
    assert r["bus_options"] == [] and r["bus_note"] is None


def test_trip_to_a_tram_place_has_notice_fallback_and_unknown_time():
    r = plan_trip_to_landmark("Al Sadd", "Place Vendome")
    assert r["status"] == "ok"
    assert r["uses_tram"] is True
    assert r["route"]["all_minutes_known"] is False
    assert r["transfer_station"] == "Legtaifiya"
    assert "Legtaifiya" in r["tram_notice"]          # tells the rider where to check
    fallback = r["fallback_route"]
    assert fallback is not None
    assert fallback["segments"][-1]["to_name"] == "Lusail QNB"


def test_fallback_route_is_metro_only_when_starting_on_the_metro():
    r = plan_trip_to_landmark("Katara", "Place Vendome")
    lines = {s["line"] for s in r["fallback_route"]["segments"]}
    assert lines <= {"RED", "GREEN", "GOLD"}


def test_starting_on_the_tram_still_works():
    r = plan_trip_to_landmark("Wadi", "Place Vendome")
    assert r["status"] == "ok"
    assert r["uses_tram"] is True
    assert r["transfer_station"] is None
    assert "before you travel" in r["tram_notice"]


def test_trip_with_a_bus_from_the_same_station_reuses_the_route():
    r = plan_trip_to_landmark("Katara", "UDST")
    assert len(r["bus_options"]) == 1
    bus = r["bus_options"][0]
    assert bus["same_station"] is True
    assert bus["route"] is r["route"]                  # same station: same route, not recomputed
    assert r["bus_note"] is not None                   # free Metrolink reminder


def test_trip_with_a_bus_from_a_different_station_gets_its_own_route():
    r = plan_trip_to_landmark("Katara", "Aberdeen")
    assert r["route"]["segments"][-1]["to_name"] == "Bin Mahmoud"      # closest station (taxi)
    bus = r["bus_options"][0]
    assert bus["same_station"] is False
    assert bus["route"]["segments"][-1]["to_name"] == "Msheireb"       # where the bus boards
    assert bus["route"]["total_stops"] == 6


def test_trip_to_a_walkable_place_still_has_bus_options():
    r = plan_trip_to_landmark("Katara", "Hyatt Plaza")
    assert len(r["bus_options"]) == 1


def test_tram_place_trip_lists_the_bus_backup_with_its_own_route():
    r = plan_trip_to_landmark("Al Sadd", "Lusail University")
    assert r["tram_notice"] is not None
    bus = r["bus_options"][0]
    assert bus["bus_number"] == "M147"
    assert bus["route"]["segments"][-1]["to_name"] == "Qatar University"
    assert bus["uses_tram"] is False


def test_trip_to_qu_gives_route_and_asks_which_building():
    r = plan_trip_to_landmark("Katara", "QU")
    assert r["status"] == "ok"
    assert r["landmark"]["status"] == "needs_detail"
    assert r["tram_notice"] is None


def test_trip_error_cases():
    assert plan_trip_to_landmark("Narnia", "Mall of Qatar")["status"] == "start_not_found"
    assert plan_trip_to_landmark("Katara", "Narnia")["status"] == "landmark_not_found"
    assert plan_trip_to_landmark("Katara", "mall")["status"] == "ambiguous_landmark"


# ---------- data checks (catch typos in your CSV files) ----------

def test_parse_meters_accepts_blanks_and_rejects_typos():
    assert parse_meters("1300", "x") == 1300
    assert parse_meters("", "x") is None
    assert parse_meters("ask_building", "x") is None
    for bad in ("200m", "12O0", "1.5"):
        try:
            parse_meters(bad, "x")
            assert False, f"'{bad}' should have been rejected"
        except ValueError:
            pass


def test_every_landmark_has_one_primary_row_and_real_stations():
    for lid in LANDMARKS:
        assert lid in LANDMARK_STATIONS, f"{lid} has no row in landmark_stations.csv"
        assert "primary" in LANDMARK_STATIONS[lid], f"{lid} has no primary row"
        for access, link in LANDMARK_STATIONS[lid].items():
            assert access in ("primary", "metro_fallback"), f"{lid}: unknown access '{access}'"
            assert link["station_id"] in STATIONS, f"{lid} points to unknown station {link['station_id']}"
    for lid in LANDMARK_STATIONS:
        assert lid in LANDMARKS, f"landmark_stations.csv has {lid}, which is not in landmarks.csv"


def test_fallback_rows_are_metro_stations_with_distances():
    for lid, links in LANDMARK_STATIONS.items():
        if "metro_fallback" in links:
            fallback = links["metro_fallback"]
            assert fallback["station_id"].startswith("M_"), f"{lid} fallback must be a metro station"
            assert fallback["walk_dist"] is not None, f"{lid} fallback has no walk_dist"


def test_parents_exist_and_every_non_campus_place_has_a_walk_distance():
    for lid, lm in LANDMARKS.items():
        if lm["parent_id"]:
            assert lm["parent_id"] in LANDMARKS, f"{lid} has unknown parent"
        has_children = any(o["parent_id"] == lid for o in LANDMARKS.values())
        if not has_children:
            assert LANDMARK_STATIONS[lid]["primary"]["walk_dist"] is not None, f"{lid} has no walk_dist"


def test_road_distance_is_blank_only_for_very_short_walks():
    # Your rule: at 300 m or less nobody drives, so road_dist is left blank.
    for lid, links in LANDMARK_STATIONS.items():
        for access, link in links.items():
            walk, road = link["walk_dist"], link["road_dist"]
            if walk is None:
                continue                                 # the campus row
            if walk <= 300:
                assert road is None, f"{lid}: walk_dist {walk} m should have no road_dist"
            else:
                assert road is not None, f"{lid}: walk_dist {walk} m needs a road_dist"


def test_bus_rows_point_to_real_places_and_metro_stations():
    for lid, buses in BUSES.items():
        assert lid in LANDMARKS, f"buses.csv has unknown landmark {lid}"
        for bus in buses:
            assert bus["bus_number"], f"{lid}: bus row with no bus_number"
            assert bus["getoff_stop"], f"{lid} bus {bus['bus_number']}: no getoff_stop"
            assert bus["from_station_id"] in STATIONS, f"{lid}: unknown station {bus['from_station_id']}"
            assert bus["from_station_id"].startswith("M_"), f"{lid}: buses board at metro stations"
            assert bus["date_checked"], f"{lid} bus {bus['bus_number']}: no date_checked"


def test_no_duplicate_bus_rows():
    seen = set()
    for lid, buses in BUSES.items():
        for bus in buses:
            key = (lid, bus["bus_number"], bus["from_station_id"])
            assert key not in seen, f"duplicate bus row {key}"
            seen.add(key)


def test_parse_flag_accepts_true_false_and_blank_but_rejects_typos():
    assert parse_flag("TRUE", "x") is True
    assert parse_flag(" false ", "x") is False
    assert parse_flag("", "x") is None
    assert parse_flag(None, "x") is None
    for bad in ("yes", "1", "TRUE;"):
        try:
            parse_flag(bad, "x")
            assert False, f"'{bad}' should have been rejected"
        except ValueError:
            pass


def test_park_and_ride_data_is_complete_once_the_column_exists():
    if not PARK_AND_RIDE:
        raise Skip("no park_and_ride column in stations.csv yet")
    metro = [sid for sid in STATIONS if sid.startswith("M_")]
    for station_id in PARK_AND_RIDE:
        assert station_id in STATIONS, f"park_and_ride has unknown station {station_id}"
        assert station_id.startswith("M_"), f"{station_id}: Park & Ride is for metro stations only"
    for station_id in metro:
        assert station_id in PARK_AND_RIDE, f"{station_id} has no TRUE/FALSE in park_and_ride"


def test_no_alias_points_to_two_different_places():
    seen = {}
    for lid, lm in LANDMARKS.items():
        for name in [lm["name"]] + lm["aliases"]:
            key = name.lower().strip()
            assert key not in seen or seen[key] == lid, f"'{name}' used by {seen[key]} and {lid}"
            seen[key] = lid


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Skip as reason:
            skipped += 1
            print(f"SKIP  {name}  ({reason})")
        except AssertionError as error:
            failed += 1
            print(f"FAIL  {name}  {error}")
    ran = len(tests) - skipped
    print(f"\n{ran - failed} of {ran} tests passed." + (f" {skipped} skipped." if skipped else ""))
