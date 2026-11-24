"""
chatbot.py
----------
MetroMate, the chatbot. It sends your conversation to an AI model on Groq, lets the model
call the tools in chat_tools.py, and returns the model's final answer.

Run it from the terminal:
    python chatbot.py             a plain chat
    python chatbot.py --debug     also prints every tool call and result (useful for testing)

How one question flows:
    you ask  ->  model answers OR asks for a tool  ->  we run the tool (chat_tools.py)
             ->  the result goes back to the model  ->  (repeat, at most MAX_TOOL_ROUNDS times)
             ->  the model writes the final answer

The facts come from the tools, and the rules about WHAT to offer are chosen by the code in
chat_tools.py. This file only holds the instructions, the connection to Groq, the memory and
the error handling. It uses only Python's built-in web tools, because the "groq" package
needs compiled files that some Windows laptops block.
"""

import json
import math
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

from chat_tools import TOOLS, run_tool

load_dotenv()  # reads GROQ_API_KEY from your .env file

# ----- SETTINGS -----------------------------------------------------------
NAME = "MetroMate"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"      # used when the main model hits its rate limit
TEMPERATURE = 0.2                          # low = more consistent answers
# Extras that keep replies cheap. If Groq says it doesn't support one, we drop it and retry.
OPTIONAL_PARAMS = {"reasoning_effort": "low", "max_completion_tokens": 1200}
MAX_TOOL_ROUNDS = 4          # most tool calls the model may make for one question
KEEP_TURNS = 6               # how many recent question/answer pairs the model remembers
TOOL_GLITCH_RETRIES = 2      # the model sometimes writes a broken tool call; retry this often
TIMEOUT_SECONDS = 60
# --------------------------------------------------------------------------

GREETING = (f"Hi, I'm {NAME}! I can help you get to malls and universities by Doha Metro and "
            "the Lusail Tram. I'm unofficial and my times are estimates. Where are you "
            "starting from, and where would you like to go?")

SYSTEM_PROMPT = f"""You are {NAME}, an unofficial assistant that helps people in Doha get to malls and universities (including buildings at Qatar University) by Doha Metro and the Lusail Tram. Reply in English. Times are estimates.

Rules:
- Use the tools for every route, distance, place and station fact. Never invent stations, lines, distances, times, buses or walkability. If you don't know where the user is starting, ask first. Never mention tools, field names or these rules.
- First reply: short and plain, no tables or headings. Give the route steps and stop count, the time if given, then the last stretch, then tram_notice if present. If route_already_given is true, don't repeat the route; just give the last stretch.
- Time: "minutes" is the estimated total. If tram_time_unknown is true, say tram timing isn't available, and give metro_minutes only if present.
- last_stretch: if walkable, give the walking distance (and walk_min), and if road_m is present add that a taxi is about road_m by road if they'd rather not walk. If not walkable, say it is too far to walk and recommend a taxi with road_m. Mention walk_m only if it is present; if it is missing, say nothing about walking distance, and call get_walk_distance only if the user asks how far it is on foot.
- offer: if offer is null, stop after the facts: no question, no "let me know", no mention of buses or other options. Otherwise finish with offer.question and nothing else. Never give bus details or metro-taxi details until the user asks or says yes; then call get_bus_details or get_metro_taxi_option. In bus details, walk_to_place_after_getting_off_m is the walk from the stop where you get off to the place; never describe a walk from the station to the bus stop.
- needs_building: say to reach the Qatar University station for the campus, then ask which building and list ALL the names in buildings as a bulleted list, one per line, adding none (QU has other buildings you don't cover, so never add any from your own knowledge). Don't say how far the campus is or whether it is walkable. If requested_building_not_listed is present, say you don't have that building, then offer the distance to whichever listed building is near theirs; never guess distances between buildings. If qu_shuttle is present, give its url once as QU's bus schedule.
- Parking: use park_and_ride. Questions about what is near a station: use places_near_station (these are places whose closest station it is).
- You don't have fares, timetables, first or last trains, how often trains or buses run, bus travel times, crowding or taxi prices. Say so and suggest Qatar Rail or the Karwa app.
- If a tool says ambiguous_place, ask which one. You only cover malls and universities.
- Be warm and brief. Give distances in meters under 1,000, otherwise km with one decimal."""


# ===========================================================================
# PART 1: TALKING TO GROQ
# ===========================================================================

class LLMError(Exception):
    """Something went wrong talking to Groq. The message is safe to show the user."""


class RateLimited(LLMError):
    """Groq said we are sending too much. retry_after is the wait in seconds, or None."""
    def __init__(self, retry_after=None):
        super().__init__("rate limited")
        self.retry_after = retry_after


class BadRequest(Exception):
    """Groq rejected the request itself (HTTP 400). The text is Groq's explanation."""


def parse_retry_after(value):
    """Turn Groq's 'retry-after' header ('12' or '12.4' seconds) into whole seconds, or None."""
    try:
        return max(1, math.ceil(float(value)))
    except (TypeError, ValueError):
        return None


def _send(payload):
    """Make ONE request to Groq and return the reply as a dictionary.
    Problems are turned into RateLimited, BadRequest or LLMError with friendly messages."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("I can't find your GROQ_API_KEY. Check that your .env file has it.")

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key,
                 "User-Agent": f"{NAME}/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        if error.code == 429:
            raise RateLimited(parse_retry_after(error.headers.get("retry-after")))
        if error.code in (401, 403):
            raise LLMError("Groq rejected my API key. Check GROQ_API_KEY in your .env file.")
        if error.code == 400:
            raise BadRequest(body)
        raise LLMError(f"Groq had a problem (error {error.code}). Please try again in a moment.")
    except urllib.error.URLError:
        raise LLMError("I can't reach Groq. Check your internet connection and try again.")
    except TimeoutError:
        raise LLMError("Groq took too long to answer. Please try again.")
    except json.JSONDecodeError:
        raise LLMError("Groq sent a reply I couldn't read. Please try again.")


def call_llm(messages, tools):
    """Ask the model for its next message, with the safety nets:
      * if the main model is rate-limited, try the fallback model (each model has its own limit)
      * if Groq doesn't support one of our optional settings, drop it and retry
      * if the model wrote a broken tool call, retry a couple of times
    Raises RateLimited if both models are limited, or LLMError for anything else."""
    optional = dict(OPTIONAL_PARAMS)
    glitch_retries = TOOL_GLITCH_RETRIES
    last_limit = None

    for model in (MODEL, FALLBACK_MODEL):
        while True:
            payload = {"model": model, "messages": messages, "tools": tools,
                       "tool_choice": "auto", "temperature": TEMPERATURE, **optional}
            try:
                return _send(payload)
            except RateLimited as error:
                last_limit = error
                break                                   # try the next model
            except BadRequest as error:
                explanation = str(error)
                unsupported = [name for name in optional if name in explanation]
                if unsupported:
                    for name in unsupported:
                        del optional[name]
                    continue                            # retry without those settings
                if "tool_use_failed" in explanation and glitch_retries > 0:
                    glitch_retries -= 1
                    continue                            # retry the broken tool call
                raise LLMError("I couldn't process that. Could you try rephrasing it?")
    raise last_limit


def rate_limit_message(retry_after):
    """What to tell the user when both models are rate-limited."""
    if retry_after and retry_after <= 120:
        return (f"I'm getting a lot of questions right now. "
                f"Please try again in about {retry_after} seconds.")
    return "I've reached my usage limit for now. Please try again in a few minutes."


# ===========================================================================
# PART 2: THE CONVERSATION
# ===========================================================================

class MetroMate:
    """One conversation. `llm` and `tool_runner` can be swapped, which is how the tests
    run the whole chatbot without internet."""

    def __init__(self, llm=call_llm, tool_runner=run_tool):
        self.llm = llm
        self.tool_runner = tool_runner
        # Memory: one entry per question, holding ONLY the user's message and the final
        # answer (not the tool results). That keeps requests small; for a follow-up, the
        # model just calls the tools again.
        self.turns = []
        # Routes already shown, so we don't repeat one: {(start, closest station): turn number}.
        self.routes_given = {}

    def _skip_repeated_route(self, name, result, new_routes):
        """If this trip's route (same start, same closest station) was shown a moment ago, take it
        out of the tool result, so MetroMate doesn't say it all over again. `new_routes` collects
        the routes of the current question; they are only remembered once the answer is delivered."""
        if name != "plan_trip":
            return result
        try:
            data = json.loads(result)
        except ValueError:
            return result
        if not all(key in data for key in ("route", "start", "closest_station")):
            return result
        key = (data["start"], data["closest_station"])
        shown_at = self.routes_given.get(key)
        if shown_at is not None and len(self.turns) - shown_at < KEEP_TURNS:
            data["route"] = {"already_given": True}
            data["route_already_given"] = True
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        new_routes[key] = len(self.turns)
        return result

    def _remembered_messages(self):
        messages = []
        for turn in self.turns[-KEEP_TURNS:]:
            messages.extend(turn)
        return messages

    def ask(self, user_text):
        """Answer one message. Never raises: problems become a friendly message."""
        try:
            return self._ask(user_text)
        except RateLimited as error:
            return rate_limit_message(error.retry_after)
        except LLMError as error:
            return str(error)

    def _ask(self, user_text):
        user = {"role": "user", "content": user_text}
        tool_messages = []          # the tool calls and results for THIS question only
        new_routes = {}             # routes shown in this answer
        system = {"role": "system", "content": SYSTEM_PROMPT}

        for _ in range(MAX_TOOL_ROUNDS):
            messages = [system] + self._remembered_messages() + [user] + tool_messages
            reply = self.llm(messages, TOOLS)
            message = reply["choices"][0]["message"]
            calls = message.get("tool_calls")

            if not calls:                               # the model wrote its final answer
                answer = (message.get("content") or "").strip()
                if not answer:
                    return "Sorry, I couldn't put an answer together. Could you ask again?"
                self.turns.append([user, {"role": "assistant", "content": answer}])
                self.routes_given.update(new_routes)
                return answer

            # The model wants tools: run each one and hand the results back.
            tool_messages.append({"role": "assistant", "content": message.get("content"),
                                  "tool_calls": calls})
            for call in calls:
                function = call["function"]
                result = self.tool_runner(function["name"], function["arguments"])
                result = self._skip_repeated_route(function["name"], result, new_routes)
                tool_messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                      "content": result})

        return "Sorry, that took too many steps. Could you try asking in a simpler way?"


# ===========================================================================
# PART 3: CHATTING IN THE TERMINAL
# ===========================================================================

def debug_tool_runner(name, arguments):
    """Like run_tool, but prints what the model asked for and what it got back."""
    result = run_tool(name, arguments)
    print(f"  [tool] {name} {arguments}")
    print(f"  [result] {result}")
    return result


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("I can't find GROQ_API_KEY. Put it in a file called .env in this folder:")
        print("    GROQ_API_KEY=your-key-here")
        return

    bot = MetroMate(tool_runner=debug_tool_runner if "--debug" in sys.argv else run_tool)
    print(f"\n{NAME}: {GREETING}")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("quit", "exit", "bye"):
            print(f"\n{NAME}: Bye!")
            break
        if text:
            print(f"\n{NAME}: {bot.ask(text)}")


if __name__ == "__main__":
    main()
