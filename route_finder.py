"""
route_finder.py finds the fastest route between two stations on the Doha Metro and Tram
network, using the CSV files in the data folder.

Every station is a "node" and every row in connections.csv is an "edge".
Dijkstra's algorithm is applied to find the path between two stations, with the least minutes.
There is a penalty of 5 minutes (estimate) each time the route changes from one line to 
another, so the finder prefers fewer changes.
"""

import csv
import heapq
import itertools
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Penalty (5 minutes) added each time a route changes from one line to another.
TRANSFER_PENALTY = 5

# ---------------------------------------------------------------------------
# PART 1: LOADING THE DATA
# ---------------------------------------------------------------------------

def read_csv(filename):
    """
    Reads a CSV file and returns a list of rows, and each row is a dictionary.
    'utf-8-sig' ignores any invisible markers made by Excel.
    """
    with open(DATA_DIR / filename, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_stations():
    """Returns a dictionary that maps station_id -> name_en (station name)."""
    return {row["station_id"]: row["name_en"] for row in read_csv("stations.csv")}


def load_line_names():
    """Returns a dictionary that maps line_id -> line_name."""
    return {row["line_id"]: row["line_name"] for row in read_csv("lines.csv")}


def load_graph():
    """
    Builds the network as a dictionary, that maps
    graph[station_id] -> [neighbour_id, minutes, line_id, minutes_known]

    Rows marked direction = "both" work in both directions, so we add the link twice.
    Rows marked direction = "forward" only work from -> to, so the link is only added once.
    """
    graph = {}
    for row in read_csv("connections.csv"):
        a = row["from_station_id"]
        b = row["to_station_id"]
        minutes = int(row["minutes"])
        line = row["line_id"]
        # The CSV stores "TRUE"/"FALSE" as text, so we compare to the text.
        known = row["minutes_known"].strip().upper() == "TRUE"

        graph.setdefault(a, []).append((b, minutes, line, known))
        if row["direction"].strip().lower() == "both":
            graph.setdefault(b, []).append((a, minutes, line, known))
    return graph


# ---------------------------------------------------------------------------
# PART 2: FINDING THE ROUTE (Dijkstra's algorithm)
# ---------------------------------------------------------------------------

def changes_line(previous_line, next_line):
    """
    True if moving from previous_line to next_line counts as changing lines.
    Metro <-> tram transfer doesn't count as changing lines, since the transfer rows
    in connections.csv already includes the walking time.
    """
    if previous_line is None:
        return False
    if previous_line == "TRANSFER" or next_line == "TRANSFER":
        return False
    return previous_line != next_line


def find_route(graph, start, end):
    """
    Returns the fastest route as a list of 'legs' (one leg is one hop between
    neighbouring stations). Returns [] if start == end, or None if no route exists.

    How it works, in simple words:
    *Keep a "to do" list of places we could be, always handling the one with the
     lowest total minutes so far (that's what heapq does for us).
    *From each place, look at every neighbouring station and add it to the list
     with the new total.
    *The first time we pull the destination off the list, we've found the fastest way.
    *We track (station, line) together, not just station, since some stations have 
    multiple lines. Changing from one line to another adds the penalty minutes.
    """
    if start == end:
        return []

    counter = itertools.count()  # tie-breaker so the heap never compares lists
    # Each item: (total_minutes_so_far, tie_breaker, station, line_we_arrived_on, legs_so_far)
    queue = [(0, next(counter), start, None, [])]
    best_cost = {}  # (station, line) -> lowest total minutes seen so far

    while queue:
        cost, _, station, line, legs = heapq.heappop(queue)

        if station == end:
            return legs

        # If we've already reached this (station, line) more cheaply, skip it.
        if best_cost.get((station, line), float("inf")) <= cost:
            continue
        best_cost[(station, line)] = cost

        for neighbour, minutes, next_line, known in graph.get(station, []):
            penalty = TRANSFER_PENALTY if changes_line(line, next_line) else 0
            leg = {
                "from": station,
                "to": neighbour,
                "line": next_line,
                "minutes": minutes,
                "known": known,
                "penalty": penalty,
            }
            heapq.heappush(
                queue,
                (cost + minutes + penalty, next(counter), neighbour, next_line, legs + [leg]),
            )

    return None  # no route found


# ---------------------------------------------------------------------------
# PART 3: TURNING THE ROUTE INTO SOMETHING READABLE
# ---------------------------------------------------------------------------

def lines_serving(graph, station_path):
    """
    Pink and Orange tram lines share some tracks. For a stretch of stations,
    return every line that serves all of its hops, e.g. ["ORANGE", "PINK"].
    """
    candidate_lines = None
    for a, b in zip(station_path, station_path[1:]):
        lines_here = {ln for (nbr, _, ln, _) in graph.get(a, []) if nbr == b}
        candidate_lines = lines_here if candidate_lines is None else candidate_lines & lines_here
    return sorted(candidate_lines or [])


def summarise_route(graph, legs, stations, line_names):
    """
    Group the legs into segments (one segment = one stretch on one line) and
    work out totals. Returns a dictionary that contains total stops, line changes, 
    known minutes, penalty minutes, and whether all minutes are known.
    """
    segments = []
    for leg in legs:
        same_line = segments and segments[-1]["line"] == leg["line"] and leg["line"] != "TRANSFER"
        if same_line:
            seg = segments[-1]
            seg["path"].append(leg["to"])
            seg["stops"] += 1
            seg["minutes"] += leg["minutes"]
        else:
            segments.append({
                "line": leg["line"],
                "path": [leg["from"], leg["to"]],
                "stops": 0 if leg["line"] == "TRANSFER" else 1,
                "minutes": leg["minutes"],
            })

    for seg in segments:
        seg["from_name"] = stations[seg["path"][0]]
        seg["to_name"] = stations[seg["path"][-1]]
        if seg["line"] == "TRANSFER":
            seg["line_names"] = []
        else:
            names = [line_names[ln] for ln in lines_serving(graph, seg["path"])]
            seg["line_names"] = names or [line_names[seg["line"]]]

    real_segments = [s for s in segments if s["line"] != "TRANSFER"]
    return {
        "segments": segments,
        "total_stops": sum(s["stops"] for s in segments),
        "line_changes": max(len(real_segments) - 1, 0),
        "known_minutes": sum(leg["minutes"] for leg in legs if leg["known"]),
        "penalty_minutes": sum(leg["penalty"] for leg in legs),
        "all_minutes_known": all(leg["known"] for leg in legs),
    }


def plural(count, word):
    """
    Adds an 's' unless the count is exactly 1, e.g. plural(1, "stop") -> "1 stop",
    plural(2, "stop") -> "2 stops", plural(0, "line change") -> "0 line changes".
    """
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def describe_route(summary):
    """Turn a summarized route into plain text. Used for testing without an LLM."""
    if not summary["segments"]:
        return "You're already there - the start and end are the same station."

    lines = []
    real_step = 0
    for seg in summary["segments"]:
        if seg["line"] == "TRANSFER":
            # Station ids starting with "M_" are metro and "T_" are tram. The walk goes
            # TO the last station of the segment, so that decides which platform.
            platform = "metro" if seg["path"][-1].startswith("M_") else "tram"
            lines.append(f"   Walk to the {platform} platform at {seg['from_name']}.")
            continue
        real_step += 1
        line_text = " or ".join(seg["line_names"])
        verb = "Take" if real_step == 1 else "Then take"
        lines.append(f"{real_step}. {verb} the {line_text} from {seg['from_name']} "
                     f"to {seg['to_name']} ({plural(seg['stops'], 'stop')}).")

    stops = plural(summary["total_stops"], "stop")
    changes = plural(summary["line_changes"], "line change")
    known = summary["known_minutes"]
    penalty = summary["penalty_minutes"]

    if summary["all_minutes_known"]:
        lines.append(f"Estimated time: about {known + penalty} min ({stops}, {changes}).")
    else:
        # Only mention the parts we actually know, so it never says "about 0 min".
        message = ""
        if known > 0:
            message = f"Known ride time: about {known} min"
            if penalty > 0:
                message += f", plus about {penalty} min for changing lines"
            message += ". "
        lines.append(message + "I couldn't find timing for the tram part.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PART 4: LOOKING UP A STATION BY NAME (used by the command line for now)
# ---------------------------------------------------------------------------

def find_station(text, stations):
    """
    Match what a person typed to a station_id, ignoring capital letters.
    Tries an exact match first, then 'contains'. If a metro and a tram station
    share a name (Legtaifiya, Lusail QNB), the metro station takes precedence.
    """
    text = text.strip().lower()
    matches = [sid for sid, name in stations.items() if name.lower() == text]
    if not matches:
        matches = [sid for sid, name in stations.items() if text in name.lower()]
    matches.sort(key=lambda sid: (not sid.startswith("M_"), sid))
    return matches[0] if matches else None


def plan_trip(start_name, end_name):
    """
    Do everything: load data, find the stations, find the route, return text.
    """
    stations = load_stations()
    line_names = load_line_names()
    graph = load_graph()

    start = find_station(start_name, stations)
    end = find_station(end_name, stations)
    if start is None:
        return f"I don't know a station called '{start_name}'."
    if end is None:
        return f"I don't know a station called '{end_name}'."

    legs = find_route(graph, start, end)
    if legs is None:
        return "I couldn't find a route between those stations."
    return describe_route(summarise_route(graph, legs, stations, line_names))


# This block only runs when you start this file directly (not when another file imports it).
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python route_finder.py "START STATION" "END STATION"')
        print('Example: python route_finder.py "Katara" "Al Riffa"')
    else:
        print(plan_trip(sys.argv[1], sys.argv[2]))
