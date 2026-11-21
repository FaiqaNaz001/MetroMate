"""
test_route_finder.py
--------------------
Simple checks that the route finder gives sensible answers.
Run with:   python test_route_finder.py

Each function starting with "test_" checks one thing. An 'assert' line says
"this must be true" - if it's false, the test fails and tells you which one.

These tests check the SHAPE of the answer (which lines, how many stops), not exact
minutes, so they keep passing if you later correct the times in the CSV.
"""

from route_finder import (
    load_graph, load_stations, load_line_names,
    find_route, summarise_route, find_station, describe_route, plural,
)

STATIONS = load_stations()
LINE_NAMES = load_line_names()
GRAPH = load_graph() 


def route(start_id, end_id):
    """Helper: find and summarise a route between two station ids."""
    legs = find_route(GRAPH, start_id, end_id)
    return summarise_route(GRAPH, legs, STATIONS, LINE_NAMES)


def lines_used(summary):
    return [s["line"] for s in summary["segments"]]


def test_same_station_means_no_trip():
    assert find_route(GRAPH, "M_KATARA", "M_KATARA") == []


def test_one_stop_on_one_line():
    r = route("M_KATARA", "M_AL_QASSAR")
    assert r["total_stops"] == 1
    assert lines_used(r) == ["RED"]
    assert r["line_changes"] == 0


def test_change_between_metro_lines():
    # Katara (Red) -> Al Riffa (Green) must change at Al Bidda: 5 + 8 stops.
    r = route("M_KATARA", "M_AL_RIFFA")
    assert lines_used(r) == ["RED", "GREEN"]
    assert r["segments"][0]["to_name"] == "Al Bidda"
    assert r["total_stops"] == 13
    assert r["line_changes"] == 1
    assert r["all_minutes_known"] is True


def test_route_works_in_both_directions():
    a = route("M_AL_AZIZIYAH", "M_AL_RIFFA")
    b = route("M_AL_RIFFA", "M_AL_AZIZIYAH")
    assert a["total_stops"] == b["total_stops"]


def test_metro_to_tram_has_transfer_and_unknown_time():
    # Al Aziziyah -> Tarfat South: Gold, Red, walk at Legtaifiya, then the tram.
    r = route("M_AL_AZIZIYAH", "T_TARFAT_SOUTH")
    assert lines_used(r)[:2] == ["GOLD", "RED"]
    assert "TRANSFER" in lines_used(r)
    assert r["all_minutes_known"] is False
    # The shared Orange/Pink stretch should list both lines.
    assert set(r["segments"][-1]["line_names"]) == {"Orange Line", "Pink Line"}


def test_one_way_tram_loop():
    # Lusail Stadium -> Lusail QNB is one stop forward, but the other way
    # round the one-way Turquoise loop means going all the way around.
    short = route("T_LUSAIL_STADIUM", "T_LUSAIL_QNB")
    long_way = route("T_LUSAIL_QNB", "T_LUSAIL_STADIUM")
    assert short["total_stops"] == 1
    assert long_way["total_stops"] == 9


def test_every_station_can_reach_every_other():
    for start in STATIONS:
        for end in STATIONS:
            assert find_route(GRAPH, start, end) is not None, f"No route {start} -> {end}"


def test_plural_helper():
    assert plural(1, "stop") == "1 stop"
    assert plural(2, "stop") == "2 stops"
    assert plural(0, "line change") == "0 line changes"


def test_wording_for_one_stop_and_no_changes():
    text = describe_route(route("M_KATARA", "M_AL_QASSAR"))
    assert "(1 stop)" in text
    assert "1 stops" not in text
    assert "0 line changes" in text


def test_wording_never_says_zero_minutes():
    # A tram-only trip has no known timing, so it must not print "about 0 min".
    text = describe_route(route("M_LEGTAIFIYA", "T_WADI"))
    assert "about 0 min" not in text
    assert "couldn't find timing" in text


def test_walk_goes_to_the_right_platform():
    to_tram = describe_route(route("M_LEGTAIFIYA", "T_WADI"))
    to_metro = describe_route(route("T_WADI", "M_KATARA"))
    assert "Walk to the tram platform" in to_tram
    assert "Walk to the metro platform" in to_metro


def test_metro_trips_never_use_the_tram():
    # The tram is still being developed and parts of it get suspended, so a trip
    # between two metro stations must never be sent through it.
    metro = [sid for sid in STATIONS if sid.startswith("M_")]
    for start in metro:
        for end in metro:
            if start != end:
                legs = find_route(GRAPH, start, end)
                used = {leg["line"] for leg in legs}
                assert used <= {"RED", "GREEN", "GOLD"}, f"{start} -> {end} uses {used}"


def test_station_name_lookup():
    assert find_station("katara", STATIONS) == "M_KATARA"
    assert find_station("  Al Bidda ", STATIONS) == "M_AL_BIDDA"
    assert find_station("Legtaifiya", STATIONS) == "M_LEGTAIFIYA"  # metro wins over tram
    assert find_station("Nowhere", STATIONS) is None


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
