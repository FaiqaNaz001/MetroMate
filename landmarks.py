"""
landmarks.py
------------
Looks up malls and universities, finds the metro/tram station closest to each,
decides how to finish the trip (walk, taxi or bus), and combines everything with
route_finder.py to plan a trip from a station to a place.

Run it from the terminal:
    python landmarks.py "Katara" "Mall of Qatar"

The data (all in the data/ folder):
  * landmarks.csv          WHAT places exist (names, aliases, category).
  * landmark_stations.csv  the CLOSEST station to each place, plus walk_dist and
                           road_dist (both in meters). A place can have a second row
                           with access = metro_fallback (nearest METRO station, for
                           places normally reached by tram, which may be suspended).
  * stations.csv           also holds an optional park_and_ride column (TRUE/FALSE) for metro stations.
  * buses.csv              Metrolink/bus options. Each row has its OWN boarding station,
                           which does not have to be the closest station.

The big idea: the data holds facts; the rules below hold the decisions.
  * Walking always comes first: if walk_dist <= WALK_LIMIT_M, the answer is "walk"
    (with the taxi's road distance mentioned for riders who would rather not walk).
  * Farther than that, the answer is "taxi" from the CLOSEST station, even when a bus
    exists. A taxi is never suggested from any other station.
  * Buses are an OFFER, never the main answer: any place with a bus listed can offer it,
    whether the place is walkable or not, and whether the bus boards at the closest
    station or a different one.
"""

import sys

from route_finder import (
    read_csv, load_stations, load_line_names, load_graph,
    find_route, summarise_route, describe_route, find_station,
)

# ----- SETTINGS -----------------------------------------------------------
WALK_LIMIT_M = 1000          # up to and including this distance: suggest walking
WALK_SPEED_M_PER_MIN = 80    # rough walking speed, used for the "about N min" estimate
MAX_OPTIONS = 5              # how many matching places to list when a name is ambiguous

# Shown whenever a route uses the tram. Tram stations open and close often, so we
# don't name specific ones; we just tell the rider to check.
TRAM_NOTICE = ("Parts of the Lusail Tram are suspended, and which stations are open "
               "keeps changing. Check tram availability {where}.")

# Metrolink routes start with "M". They are free, but riders must still validate.
METROLINK_NOTE = ("Metrolink buses are free, but you still need to tap in with a card "
                  "or scan a QR ticket.")
# --------------------------------------------------------------------------


# ===========================================================================
# PART 1: LOADING THE DATA
# ===========================================================================

def clean(row, column):
    """Read one cell as stripped text. Missing cells become ''."""
    return (row.get(column) or "").strip()


def parse_meters(text, where):
    """Turn a CSV cell into whole meters. Blank (or the 'ask_building' marker used for
    the Qatar University campus row) becomes None. Anything else that isn't a whole
    number is a typo, so we stop with a clear message instead of hiding it."""
    text = (text or "").strip()
    if text in ("", "ask_building"):
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"Expected a whole number of meters but found '{text}' ({where})")


def parse_flag(text, where):
    """Turn a TRUE/FALSE cell into True/False. Blank becomes None (unknown, or not
    applicable). Anything else is a typo, so we stop with a clear message."""
    text = (text or "").strip().upper()
    if text == "":
        return None
    if text in ("TRUE", "FALSE"):
        return text == "TRUE"
    raise ValueError(f"Expected TRUE or FALSE but found '{text}' ({where})")


def load_park_and_ride():
    """Return {station_id: True/False} for every station whose park_and_ride cell in
    stations.csv is filled in. The column is optional: if it doesn't exist yet, or a
    station's cell is blank (for example tram stations), that station is simply left out."""
    flags = {}
    for row in read_csv("stations.csv"):
        flag = parse_flag(row.get("park_and_ride"), f"{row['station_id']} park_and_ride")
        if flag is not None:
            flags[row["station_id"]] = flag
    return flags


def load_landmarks():
    """Return {landmark_id: {"id", "name", "aliases", "category", "parent_id"}}."""
    landmarks = {}
    for row in read_csv("landmarks.csv"):
        landmarks[row["landmark_id"]] = {
            "id": row["landmark_id"],
            "name": row["name"],
            # "MOQ; Mall of Qatar Doha" becomes ["MOQ", "Mall of Qatar Doha"]
            "aliases": [a.strip() for a in row["aliases"].split(";") if a.strip()],
            "category": row["category"],
            "parent_id": clean(row, "parent_id") or None,  # blank means "no parent"
        }
    return landmarks


def load_landmark_stations():
    """Return {landmark_id: {access: {"station_id", "walk_dist", "road_dist"}}}.

    "access" is "primary" or "metro_fallback", so one place can hold two entries.
    walk_dist / road_dist are in meters, or None when blank (road_dist is left blank
    for very short walks, and both are blank for the Qatar University campus row).
    """
    result = {}
    for row in read_csv("landmark_stations.csv"):
        lid = row["landmark_id"]
        access = clean(row, "access") or "primary"
        result.setdefault(lid, {})[access] = {
            "station_id": row["station_id"],
            "walk_dist": parse_meters(row["walk_dist"], f"{lid} walk_dist"),
            "road_dist": parse_meters(row["road_dist"], f"{lid} road_dist"),
        }
    return result


def load_buses():
    """Return {landmark_id: [bus, bus, ...]} from buses.csv.
    Rows with no landmark_id are skipped: Excel often saves a CSV with a huge number
    of empty rows at the bottom."""
    buses = {}
    for row in read_csv("buses.csv"):
        lid = clean(row, "landmark_id")
        if not lid:
            continue
        buses.setdefault(lid, []).append({
            "bus_number": clean(row, "bus_number"),
            "from_station_id": clean(row, "from_station_id"),
            "getoff_stop": clean(row, "getoff_stop"),
            "stop_to_landmark_m": parse_meters(row.get("dist_from_landmark"),
                                               f"{lid} dist_from_landmark"),
            "source": clean(row, "source"),
            "date_checked": clean(row, "date_checked"),
        })
    return buses


# Load everything once when this file is imported.
STATIONS = load_stations()
LINE_NAMES = load_line_names()
GRAPH = load_graph()
LANDMARKS = load_landmarks()
LANDMARK_STATIONS = load_landmark_stations()
BUSES = load_buses()
PARK_AND_RIDE = load_park_and_ride()   # {station_id: True/False}, empty until the column exists
# {"RED": "metro", "PINK": "tram", ...} - lets us tell whether a route uses the tram.
LINE_NETWORKS = {row["line_id"]: row["network"] for row in read_csv("lines.csv")}


# ===========================================================================
# PART 2: FINDING A LANDMARK FROM WHAT THE USER TYPED
# ===========================================================================

# Words that carry no meaning when matching a place ("the research complex at QU").
FILLER_WORDS = {"the", "a", "an", "at", "in", "of", "to", "and", "building", "campus"}
# Words that name a whole CATEGORY of places. Typed alone, they must lead to "which one?"
GENERIC_WORDS = {"mall", "university", "college", "center", "centre"}


def normalise(text):
    """Lower-case, turn '&' into 'and', drop punctuation and tidy spaces, so that
    'Sports & Events Complex!' becomes 'sports and events complex'."""
    text = text.lower().replace("&", " and ")
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def meaningful_words(text):
    """The set of words that matter in a name, ignoring filler words and word order."""
    return {word for word in normalise(text).split() if word not in FILLER_WORDS}


def all_names(landmark):
    """Every name a place can be called: its official name plus its aliases."""
    return [normalise(landmark["name"])] + [normalise(a) for a in landmark["aliases"]]


def find_landmark(text, landmarks):
    """Match typed text to landmarks. Returns (status, list_of_ids) where status is:
        "found"      - exactly one place matches
        "ambiguous"  - several places match (e.g. just "mall")
        "not_found"  - nothing matches

    Matching works on WORDS, not letters, so "male dorm" can't match "female dorm". It tries
    four things in order and stops at the first that finds something:
      0. exactly a name or alias, ignoring capitals and punctuation ("the mall" = "The Mall")
      1. the same meaningful words as a name or alias, in any order ("law college" = "College of Law")
      2. the typed words all appear inside a name ("research" is inside "Research Complex")
      3. the typed text contains a whole name ("the research complex at the main campus"),
         using the longest name found
    A bare category word like "mall" skips step 1, so it leads to a "which one?" list.
    """
    query = normalise(text)
    words = meaningful_words(text)
    if not words:
        return "not_found", []

    def verdict(ids):
        return ("found", ids) if len(ids) == 1 else ("ambiguous", ids)

    same_text = [lid for lid, lm in landmarks.items() if query in all_names(lm)]
    if same_text:
        return verdict(same_text)

    if not words <= GENERIC_WORDS:
        same_words = [lid for lid, lm in landmarks.items()
                      if any(meaningful_words(name) == words for name in all_names(lm))]
        if same_words:
            return verdict(same_words)

    if len(query) >= 3:        # too-short text like "al" would match far too much
        inside = [lid for lid, lm in landmarks.items()
                  if any(words <= meaningful_words(name) for name in all_names(lm))]
        if inside:
            return verdict(inside)

    longest = {}               # landmark_id -> number of words in its longest name found
    for lid, lm in landmarks.items():
        for name in all_names(lm):
            name_words = meaningful_words(name)
            if len(name_words) >= 2 and name_words <= words:
                longest[lid] = max(longest.get(lid, 0), len(name_words))
    if longest:
        best = max(longest.values())
        return verdict([lid for lid, size in longest.items() if size == best])

    return "not_found", []


# ===========================================================================
# PART 3: HOW TO FINISH THE TRIP (walk / taxi / bus rules)
# ===========================================================================

def transport_suggestion(walk_dist):
    """Walk if the walk is WALK_LIMIT_M or less; otherwise taxi."""
    return "walk" if walk_dist <= WALK_LIMIT_M else "taxi"


def distance_details(link):
    """Turn one CSV link ({"walk_dist", "road_dist", ...}) into the facts about the
    last stretch: both distances, walk or taxi, and a rough walking time."""
    walk = link["walk_dist"]
    return {
        "walk_dist": walk,
        "road_dist": link["road_dist"],
        "suggestion": transport_suggestion(walk),
        "walk_minutes": round(walk / WALK_SPEED_M_PER_MIN),
    }


def bus_options(landmark_id, closest_station_id):
    """All buses listed for a place. same_station is True when the bus boards at the
    place's closest station, False when it boards somewhere else."""
    options = []
    for bus in BUSES.get(landmark_id, []):
        board = bus["from_station_id"]
        options.append({
            "bus_number": bus["bus_number"],
            "from_station_id": board,
            "from_station_name": STATIONS[board],
            "getoff_stop": bus["getoff_stop"],
            "stop_to_landmark_m": bus["stop_to_landmark_m"],
            "same_station": board == closest_station_id,
            "is_metrolink": bus["bus_number"].upper().startswith("M"),
            "source": bus["source"],
            "date_checked": bus["date_checked"],
        })
    return options


def get_landmark_info(landmark_id):
    """Everything the chatbot needs to know about one place.
    Extra keys: "buses" (every bus listed for the place, possibly empty) and, for tram places,
    "fallback" (the nearest metro station)."""
    landmark = LANDMARKS[landmark_id]
    links = LANDMARK_STATIONS[landmark_id]
    primary = links["primary"]

    info = {
        "landmark_id": landmark_id,
        "name": landmark["name"],
        "category": landmark["category"],
        "station_id": primary["station_id"],
        "station_name": STATIONS[primary["station_id"]],
    }

    walk = primary["walk_dist"]
    if walk is None:
        # A big place with no single distance (the QU campus): list its sub-places.
        info["status"] = "needs_detail"
        info["options"] = [lm["name"] for lm in LANDMARKS.values()
                           if lm["parent_id"] == landmark_id]
    else:
        info["status"] = "found"
        info.update(distance_details(primary))

    info["buses"] = bus_options(landmark_id, primary["station_id"])

    if "metro_fallback" in links:
        fallback = links["metro_fallback"]
        info["fallback"] = {
            "station_id": fallback["station_id"],
            "station_name": STATIONS[fallback["station_id"]],
        }
        info["fallback"].update(distance_details(fallback))
    return info


# ===========================================================================
# PART 4: THE TRAM NOTICE
# ===========================================================================

def route_uses_tram(summary):
    """True if any part of the route is on a tram line."""
    return any(LINE_NETWORKS.get(seg["line"]) == "tram" for seg in summary["segments"])


def tram_transfer_station(summary):
    """Name of the metro station where the route switches to the tram, or None
    (for example when the rider starts on the tram already)."""
    for seg in summary["segments"]:
        # A TRANSFER segment whose last station starts with "T_" goes metro -> tram.
        if seg["line"] == "TRANSFER" and seg["path"][-1].startswith("T_"):
            return seg["from_name"]
    return None


def tram_notice(transfer_station):
    """The 'check tram availability' message, mentioning the metro station if known."""
    where = f"once you reach {transfer_station} station" if transfer_station else "before you travel"
    return TRAM_NOTICE.format(where=where)


# ===========================================================================
# PART 5: PLANNING A WHOLE TRIP (route + last stretch + bus options)
# ===========================================================================

def plan_trip_to_landmark(start_text, landmark_text):
    """Plan a trip from a station to a place. Returns a dictionary whose "status" is:
        "ok"                  - route and landmark info included
        "start_not_found"     - we don't know that start station
        "landmark_not_found"  - we don't know that place
        "ambiguous_landmark"  - several places match; "options" lists them
        "no_route"            - (shouldn't happen) no way to get there

    When status is "ok" the dictionary also has:
        "route"            the route to the place's closest station
        "landmark"         facts about the place (distances, fallback, buses)
        "uses_tram"        True if the route includes the tram
        "transfer_station" metro station where the rider switches to the tram, or None
        "tram_notice"      the 'check tram availability' text, or None if no tram
        "fallback_route"   route to the nearest metro station, or None
        "bus_options"      every bus listed for the place, each with its own "route" to the
                           bus's boarding station (empty list if the place has no bus)
        "bus_note"         the free-Metrolink text, or None
    The chatbot will call this and put the result into friendly words.
    """
    start = find_station(start_text, STATIONS)
    if start is None:
        return {"status": "start_not_found"}

    status, ids = find_landmark(landmark_text, LANDMARKS)
    if status == "not_found":
        return {"status": "landmark_not_found"}
    if status == "ambiguous":
        names = [LANDMARKS[i]["name"] for i in ids][:MAX_OPTIONS]
        return {"status": "ambiguous_landmark", "options": names}

    info = get_landmark_info(ids[0])
    legs = find_route(GRAPH, start, info["station_id"])
    if legs is None:
        return {"status": "no_route"}

    route = summarise_route(GRAPH, legs, STATIONS, LINE_NAMES)
    uses_tram = route_uses_tram(route)
    transfer_station = tram_transfer_station(route)

    # If the place also has a metro-only option, plan that route too.
    fallback_route = None
    if "fallback" in info:
        fallback_legs = find_route(GRAPH, start, info["fallback"]["station_id"])
        if fallback_legs is not None:
            fallback_route = summarise_route(GRAPH, fallback_legs, STATIONS, LINE_NAMES)

    # Buses: each one boards at its own station, so each gets its own route.
    bus_list = []
    if info["buses"]:
        routes_by_station = {info["station_id"]: route}   # reuse the route we already have
        for bus in info["buses"]:
            board = bus["from_station_id"]
            if board not in routes_by_station:
                bus_legs = find_route(GRAPH, start, board)
                routes_by_station[board] = (summarise_route(GRAPH, bus_legs, STATIONS, LINE_NAMES)
                                            if bus_legs is not None else None)
            bus_route = routes_by_station[board]
            if bus_route is None:
                continue
            option = dict(bus)
            option["route"] = bus_route
            option["uses_tram"] = route_uses_tram(bus_route)
            option["tram_notice"] = (tram_notice(tram_transfer_station(bus_route))
                                     if option["uses_tram"] else None)
            bus_list.append(option)

    return {
        "status": "ok",
        "start_station": STATIONS[start],
        "route": route,
        "landmark": info,
        "uses_tram": uses_tram,
        "transfer_station": transfer_station,
        "tram_notice": tram_notice(transfer_station) if uses_tram else None,
        "fallback_route": fallback_route,
        "bus_options": bus_list,
        "bus_note": METROLINK_NOTE if any(b["is_metrolink"] for b in bus_list) else None,
    }


# ===========================================================================
# PART 6: PLAIN-TEXT VERSION (for testing in the terminal; the chatbot won't use it)
# ===========================================================================

def format_distance(meters):
    """850 -> '850 m', 4900 -> '4.9 km'."""
    return f"{meters} m" if meters < 1000 else f"{meters / 1000:.1f} km"


def last_stretch_text(place, details, mention_bus_check=True):
    """One sentence about getting from a station to the place. `details` needs
    station_name, walk_dist, suggestion, walk_minutes; road_dist is optional."""
    walk = format_distance(details["walk_dist"])
    station = details["station_name"]
    if details["suggestion"] == "walk":
        text = (f"{place} is about {walk} from {station} station "
                f"(roughly {details['walk_minutes']} min on foot), so walking is fine.")
        road = details.get("road_dist")
        if road:
            text += f" If you're not comfortable walking, a taxi is about {format_distance(road)} by road."
        return text

    bus_note = (" I don't have bus information for this place, so check the Karwa app."
                if mention_bus_check else "")
    road = details.get("road_dist")
    if road:
        return (f"{place} is too far to walk from {station} station (about {walk} on foot). "
                f"By road it's about {format_distance(road)}, so a taxi is probably easiest."
                + bus_note)
    return (f"{place} is about {walk} on foot from {station} station, which is too far "
            f"to walk comfortably. A taxi is probably easiest." + bus_note)


def bus_text(place, bus):
    """One sentence about a bus: which one, where to board, where to get off."""
    kind = "Metrolink bus" if bus["is_metrolink"] else "bus"
    text = (f"Take {kind} {bus['bus_number']} from {bus['from_station_name']} station "
            f"and get off at stop {bus['getoff_stop']}")
    if bus["stop_to_landmark_m"]:
        text += f" ({format_distance(bus['stop_to_landmark_m'])} from {place})"
    text += "."
    if bus["date_checked"]:
        text += f" (route checked {bus['date_checked']})"
    return text


def describe_trip(result):
    """Turn a plan_trip_to_landmark result into plain text."""
    status = result["status"]
    if status == "start_not_found":
        return "I don't know that starting station."
    if status == "landmark_not_found":
        return "I don't have that place yet. I cover malls and universities."
    if status == "ambiguous_landmark":
        return "Which one do you mean? " + "; ".join(result["options"])
    if status == "no_route":
        return "I couldn't find a route."

    info = result["landmark"]
    place = info["name"]
    lines = [f"From {result['start_station']} to {place}:",
             describe_route(result["route"])]

    if info["status"] == "needs_detail":
        lines.append(f"{place} is large. Which part are you heading to? "
                     + "; ".join(info["options"]))
    else:
        lines.append(last_stretch_text(place, info,
                                       mention_bus_check=not result["bus_options"]))

    if result["tram_notice"]:
        lines.append("Note: " + result["tram_notice"])

    fallback = info.get("fallback")
    if fallback and result["fallback_route"] is not None:
        lines.append("")
        lines.append(f"If the tram isn't running, go to {fallback['station_name']} "
                     "metro station instead:")
        lines.append(describe_route(result["fallback_route"]))
        lines.append(last_stretch_text(place, fallback, mention_bus_check=False))

    for bus in result["bus_options"]:
        lines.append("")
        if bus["same_station"]:
            lines.append("You can also take a bus from the same station:")
        else:
            lines.append(f"Alternatively, go to {bus['from_station_name']} station instead:")
            lines.append(describe_route(bus["route"]))
        lines.append(bus_text(place, bus))
        if bus["tram_notice"]:
            lines.append("Note: " + bus["tram_notice"])
    if result["bus_note"]:
        lines.append(result["bus_note"])
    return "\n".join(lines)


# This block only runs when you start this file directly.
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python landmarks.py "START STATION" "PLACE"')
        print('Example: python landmarks.py "Katara" "Mall of Qatar"')
    else:
        print(describe_trip(plan_trip_to_landmark(sys.argv[1], sys.argv[2])))
