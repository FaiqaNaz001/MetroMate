"""
chat_tools.py
-------------
The "tools" the chatbot's AI model can call. Each tool is a normal Python function that
takes plain text (station and place names) and returns a SMALL dictionary of facts.

Three design rules:
  1. Small results. The free Groq plan has a tokens-per-minute budget, so every result
     contains only what the AI needs to write its reply.
  2. Details are hidden by design. plan_trip never contains bus details or the taxi-from-
     metro option. Those only come back from get_bus_details / get_metro_taxi_option,
     which the AI calls only when the user asks.
  3. The CODE chooses the one offer to make (result["offer"]["question"]). The AI only
     words it, so it can't forget a rule or offer three things at once.

The facts come from landmarks.py and route_finder.py, which are already tested.
"""

import json

from route_finder import find_station, find_route, summarise_route, plural
from landmarks import (
    STATIONS, LANDMARKS, LANDMARK_STATIONS, GRAPH, LINE_NAMES, PARK_AND_RIDE,
    WALK_LIMIT_M, MAX_OPTIONS,
    plan_trip_to_landmark, tram_notice, route_uses_tram, tram_transfer_station, normalise,
)

# ----- SETTINGS -----------------------------------------------------------
# Link to Qatar University's published bus schedule. If it is empty, the chatbot simply
# doesn't mention the shuttle. QU may replace this file, so recheck the link from time to time.
QU_SHUTTLE_URL = "https://www.qu.edu.qa/sites/default/files/2026-11/Bus%20Schedules_0.pdf"

QU_CAMPUS_ID = "UNI17"      # the landmark_id of the Qatar University campus row
MAX_NEAR = 8                # how many places places_near_station lists at most
# --------------------------------------------------------------------------


# ===========================================================================
# PART 1: BUILDING SMALL RESULTS
# ===========================================================================

def compact_route(summary):
    """Shrink a route summary to what the AI needs: readable steps, stop count, minutes.
    "minutes" is only given when every leg's time is known (metro-only trips). If part of
    the trip is on the tram, only "metro_minutes" is given and tram_time_unknown is True."""
    if not summary["segments"]:
        return {"already_there": True}

    steps = []
    for seg in summary["segments"]:
        if seg["line"] == "TRANSFER":
            platform = "metro" if seg["path"][-1].startswith("M_") else "tram"
            steps.append(f"Walk to the {platform} platform at {seg['from_name']}")
        else:
            lines = " or ".join(seg["line_names"])
            steps.append(f"{lines}: {seg['from_name']} to {seg['to_name']} "
                         f"({plural(seg['stops'], 'stop')})")

    route = {"steps": steps,
             "total_stops": summary["total_stops"],
             "line_changes": summary["line_changes"]}
    minutes = summary["known_minutes"] + summary["penalty_minutes"]
    if summary["all_minutes_known"]:
        route["minutes"] = minutes
    else:
        route["tram_time_unknown"] = True
        if summary["known_minutes"] > 0:
            route["metro_minutes"] = minutes
    return route


def last_stretch(info):
    """The facts about getting from the closest station to the place."""
    walkable = info["walk_dist"] <= WALK_LIMIT_M
    stretch = {
        "from_station": info["station_name"],
        "station_type": "tram" if info["station_id"].startswith("T_") else "metro",
        "walk_m": info["walk_dist"],
        "walkable": walkable,
        "advice": "walk" if walkable else "taxi",
    }
    if walkable:
        stretch["walk_min"] = info["walk_minutes"]
    if info["road_dist"]:           # blank for very short walks, so no taxi line for those
        stretch["road_m"] = info["road_dist"]
    return stretch


def choose_offer(info):
    """Pick the ONE thing the chatbot offers at the end of its first reply (or None).
    Bus details and the taxi-from-metro option stay hidden until the user says yes."""
    buses = info["buses"]
    fallback = info.get("fallback")
    bus_kind = "free bus service" if any(b["is_metrolink"] for b in buses) else "bus service"
    closest_is_tram = info["station_id"].startswith("T_")

    if closest_is_tram:
        if fallback and buses:
            return {"type": "tram_alternatives", "question": (
                f"Would you like the road distance from {fallback['station_name']} metro "
                f"station for a taxi, or details of a {bus_kind} instead?")}
        if fallback:
            return {"type": "tram_taxi", "question": (
                f"Would you like to know the road distance from {fallback['station_name']} "
                "metro station, in case you take a taxi?")}
        if buses:
            return {"type": "bus_other_station",
                    "question": f"There's also a {bus_kind}. Want the details?"}
        return None

    if buses:
        if any(b["same_station"] for b in buses):
            return {"type": "bus_same_station", "question": (
                f"There's also a {bus_kind} from {info['station_name']} station. "
                "Want the details?")}
        return {"type": "bus_other_station", "question": (
            f"There's also a {bus_kind} from a different station. Want the details?")}
    return None


def is_at_qatar_university(landmark_id):
    """True for the QU campus row and for every building whose parent is the campus."""
    return landmark_id == QU_CAMPUS_ID or LANDMARKS[landmark_id]["parent_id"] == QU_CAMPUS_ID


def shuttle_info():
    """The QU shuttle pointer, or None until a link has been pasted into QU_SHUTTLE_URL."""
    if not QU_SHUTTLE_URL:
        return None
    return {"url": QU_SHUTTLE_URL,
            "note": ("Qatar University also publishes a bus schedule. Give this link once, as QU's "
                     "bus schedule. Don't describe what it contains or say anything else about "
                     "the buses.")}


def mentions_qu(text):
    """True if the typed place mentions Qatar University ('the pharmacy college at QU')."""
    words = normalise(text).split()
    return "qu" in words or "qatar university" in " ".join(words)


def trip_error(result):
    """Turn a failed plan_trip_to_landmark status into a small error dictionary (or None)."""
    status = result["status"]
    if status == "start_not_found":
        return {"error": "start_not_found",
                "hint": "Ask the user which metro or tram station they are starting from."}
    if status == "landmark_not_found":
        return {"error": "place_not_found",
                "hint": ("I only cover malls and universities. If the user means a building at "
                         "Qatar University, call plan_trip again with place='Qatar University' "
                         "to get the list of buildings.")}
    if status == "ambiguous_landmark":
        return {"error": "ambiguous_place", "options": result["options"]}
    if status == "no_route":
        return {"error": "no_route"}
    return None


# ===========================================================================
# PART 2: THE TOOLS
# ===========================================================================

def plan_trip(start, place):
    """First reply for a trip: route, last stretch, tram notice, and the one offer."""
    result = plan_trip_to_landmark(start, place)

    # A building at Qatar University that isn't in the data: give the route to the campus and
    # the list of buildings we DO have, so the AI never has to guess or list them from memory.
    if result["status"] == "landmark_not_found" and mentions_qu(place):
        campus = plan_trip(start, LANDMARKS[QU_CAMPUS_ID]["name"])
        if "error" not in campus:
            campus["requested_building_not_listed"] = place
        return campus

    error = trip_error(result)
    if error:
        return error

    info = result["landmark"]
    out = {
        "place": info["name"],
        "start": result["start_station"],
        "closest_station": info["station_name"],
        "route": compact_route(result["route"]),
    }

    if is_at_qatar_university(info["landmark_id"]) and shuttle_info():
        out["qu_shuttle"] = shuttle_info()

    if info["status"] == "needs_detail":          # the QU campus: ask which building
        out["needs_building"] = True
        out["building_count"] = len(info["options"])
        out["buildings"] = info["options"]
        return out

    out["last_stretch"] = last_stretch(info)
    if result["tram_notice"]:
        out["tram_notice"] = result["tram_notice"]
    offer = choose_offer(info)
    if offer:
        out["offer"] = offer
    return out


def get_bus_details(start, place):
    """Bus details. The AI calls this only after the user asks about or accepts a bus offer."""
    result = plan_trip_to_landmark(start, place)
    error = trip_error(result)
    if error:
        return error
    if not result["bus_options"]:
        return {"error": "no_bus_info",
                "hint": "No bus is listed for this place. Don't suggest one."}

    buses = []
    for bus in result["bus_options"]:
        details = {
            "bus": bus["bus_number"] + (" (Metrolink, free)" if bus["is_metrolink"] else ""),
            "board_at": f"{bus['from_station_name']} station",
            "get_off_at_stop": bus["getoff_stop"],
            "same_station_as_closest": bus["same_station"],
            "route_checked": bus["date_checked"],
        }
        if bus["stop_to_landmark_m"]:
            details["stop_to_place_m"] = bus["stop_to_landmark_m"]
        if not bus["same_station"]:     # a different boarding station needs its own route
            details["route_to_boarding_station"] = compact_route(bus["route"])
        if bus["tram_notice"]:
            details["tram_notice"] = bus["tram_notice"]
        buses.append(details)

    out = {"place": result["landmark"]["name"], "buses": buses,
           "timing": ("No bus timing is available because it varies with traffic. "
                      "The Karwa app shows live arrivals.")}
    if result["bus_note"]:
        out["tap_in_note"] = result["bus_note"]
    return out


def get_metro_taxi_option(start, place):
    """For tram places: the nearest METRO station, the route to it and the taxi distance.
    The AI calls this only after the user accepts the tram offer."""
    result = plan_trip_to_landmark(start, place)
    error = trip_error(result)
    if error:
        return error
    fallback = result["landmark"].get("fallback")
    if not fallback or result["fallback_route"] is None:
        return {"error": "no_metro_option",
                "hint": "This place has no separate metro-and-taxi option."}
    return {
        "place": result["landmark"]["name"],
        "metro_station": fallback["station_name"],
        "route": compact_route(result["fallback_route"]),
        "walk_m": fallback["walk_dist"],
        "road_m": fallback["road_dist"],
        "advice": fallback["suggestion"],
    }


def places_near_station(station):
    """Places whose CLOSEST station is the given one, nearest first."""
    station_id = find_station(station, STATIONS)
    if station_id is None:
        return {"error": "station_not_found"}

    places = []
    for landmark_id, links in LANDMARK_STATIONS.items():
        link = links["primary"]
        if link["station_id"] == station_id and link["walk_dist"] is not None:
            places.append({"name": LANDMARKS[landmark_id]["name"], "walk_m": link["walk_dist"]})
    places.sort(key=lambda place: place["walk_m"])

    out = {"station": STATIONS[station_id], "count": len(places), "places": places[:MAX_NEAR]}
    if len(places) > MAX_NEAR:
        out["more"] = len(places) - MAX_NEAR
    if not places:
        out["note"] = "None of the places in my data have this as their closest station."
    return out


def park_and_ride(station):
    """Does a metro station have Park & Ride? Pass "all" to list the stations that do."""
    if not PARK_AND_RIDE:
        return {"error": "no_park_and_ride_data",
                "hint": "I don't have Park & Ride information. Suggest checking Qatar Rail."}

    if station.strip().lower() in ("", "all", "any"):
        names = sorted(STATIONS[sid] for sid, has_it in PARK_AND_RIDE.items() if has_it)
        return {"stations_with_park_and_ride": names}

    station_id = find_station(station, STATIONS)
    if station_id is None:
        return {"error": "station_not_found"}
    out = {"station": STATIONS[station_id]}
    has_it = PARK_AND_RIDE.get(station_id)
    if has_it is None:                      # blank in the data, for example tram stations
        out["park_and_ride"] = "unknown"
        out["note"] = "I only have Park & Ride information for metro stations."
    else:
        out["park_and_ride"] = has_it
    return out


def plan_route(start, end):
    """A route between two stations, when the user isn't asking about a place."""
    start_id = find_station(start, STATIONS)
    end_id = find_station(end, STATIONS)
    if start_id is None:
        return {"error": "start_not_found",
                "hint": "Ask the user which station they are starting from."}
    if end_id is None:
        return {"error": "end_not_found", "hint": "Ask the user for the destination station."}

    legs = find_route(GRAPH, start_id, end_id)
    if legs is None:
        return {"error": "no_route"}
    summary = summarise_route(GRAPH, legs, STATIONS, LINE_NAMES)
    out = {"start": STATIONS[start_id], "end": STATIONS[end_id], "route": compact_route(summary)}
    if route_uses_tram(summary):
        out["tram_notice"] = tram_notice(tram_transfer_station(summary))
    return out


# ===========================================================================
# PART 3: TELLING THE AI WHAT TOOLS EXIST
# ===========================================================================

def _tool(name, description, properties):
    """Build one tool description in the format the Groq API expects."""
    return {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": "string", "description": text}
                           for key, text in properties.items()},
            "required": list(properties),
        },
    }}


START = "Station the user starts from, as they wrote it"
PLACE = "Mall, university or Qatar University building"

TOOLS = [
    _tool("plan_trip",
          "Plan a trip from a station to a place. Returns the route, walk/taxi facts, any "
          "tram notice and the one offer to make.",
          {"start": START, "place": PLACE}),
    _tool("get_bus_details",
          "Bus details for a place. Call ONLY after the user asks about or accepts a bus offer.",
          {"start": START, "place": PLACE}),
    _tool("get_metro_taxi_option",
          "Nearest metro station, route and taxi distance for a tram place. Call ONLY after "
          "the user accepts that offer.",
          {"start": START, "place": PLACE}),
    _tool("places_near_station",
          "Places whose closest station is the given station.",
          {"station": "Station name"}),
    _tool("park_and_ride",
          "Whether a metro station has Park & Ride. Use 'all' to list the stations that do.",
          {"station": "Station name, or 'all'"}),
    _tool("plan_route",
          "Route between two stations when no place is involved.",
          {"start": "Start station", "end": "End station"}),
]

TOOL_FUNCTIONS = {
    "plan_trip": plan_trip,
    "get_bus_details": get_bus_details,
    "get_metro_taxi_option": get_metro_taxi_option,
    "places_near_station": places_near_station,
    "park_and_ride": park_and_ride,
    "plan_route": plan_route,
}


def run_tool(name, arguments):
    """Run one tool by name and return its result as a compact JSON string.
    `arguments` is what the AI sent: a JSON string like '{"start": "Katara", ...}'.
    A bad tool name or bad arguments become a small error result, never a crash."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        result = {"error": "unknown_tool"}
    else:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            result = function(**args)
        except (TypeError, ValueError) as error:      # bad JSON or missing/extra arguments
            result = {"error": "bad_arguments", "detail": str(error)}
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
