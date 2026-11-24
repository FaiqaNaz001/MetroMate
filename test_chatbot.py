"""
test_chatbot.py
---------------
Checks MetroMate's plumbing without internet or an API key. Run with:
    python test_chatbot.py

Two helpers do the pretending:
  * FakeLLM plays the AI model: it returns pre-written replies and records what it was sent.
  * A tiny local web server plays Groq, so the real connection code (_send) gets tested too.

What these tests CANNOT check is how the real model words its answers. That is what you
check by hand, by chatting with it.
"""

import http.server
import json
import os
import threading

import chatbot
from chatbot import (
    MetroMate, LLMError, RateLimited, BadRequest, call_llm, parse_retry_after,
    rate_limit_message, SYSTEM_PROMPT, GREETING, KEEP_TURNS, MAX_TOOL_ROUNDS,
    MODEL, FALLBACK_MODEL,
)
from chat_tools import TOOLS


# ---------- pretend model ----------

def tool_call(name, arguments, call_id="call_1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


def reply(content=None, tool_calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": tool_calls}}]}


class FakeLLM:
    """Returns the scripted replies one by one and remembers every request it received."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def roles(messages):
    return [m["role"] for m in messages]


# ---------- the conversation loop ----------

def test_answer_without_tools():
    llm = FakeLLM([reply("Hello! Where are you starting from?")])
    bot = MetroMate(llm=llm)
    assert bot.ask("hi") == "Hello! Where are you starting from?"
    sent = llm.calls[0]
    assert roles(sent["messages"]) == ["system", "user"]
    assert sent["messages"][0]["content"] == SYSTEM_PROMPT
    assert sent["tools"] == TOOLS


def test_tool_round_trip():
    llm = FakeLLM([
        reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "Mall of Qatar"})]),
        reply("Take the Red Line, then the Green Line."),
    ])
    bot = MetroMate(llm=llm)
    assert bot.ask("How do I get to Mall of Qatar from Katara?") == "Take the Red Line, then the Green Line."
    assert len(llm.calls) == 2
    second = llm.calls[1]["messages"]
    assert roles(second) == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["id"] == "call_1"
    assert second[3]["tool_call_id"] == "call_1"
    result = json.loads(second[3]["content"])
    assert result["place"] == "Mall of Qatar" and "route" in result      # the real tool really ran


def test_several_tool_calls_in_one_reply():
    llm = FakeLLM([
        reply(tool_calls=[tool_call("park_and_ride", {"station": "Katara"}, "a"),
                          tool_call("places_near_station", {"station": "Katara"}, "b")]),
        reply("Done."),
    ])
    MetroMate(llm=llm).ask("parking and places?")
    tool_messages = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["a", "b"]


def test_bad_tool_calls_are_passed_back_not_crashed():
    llm = FakeLLM([reply(tool_calls=[tool_call("nope", {})]), reply("Sorry about that.")])
    assert MetroMate(llm=llm).ask("x") == "Sorry about that."
    tool_message = llm.calls[1]["messages"][-1]
    assert json.loads(tool_message["content"])["error"] == "unknown_tool"


def test_hidden_details_flow_through_the_loop():
    # Question 1 gets the short answer; "yes" makes the model fetch the bus details.
    llm = FakeLLM([
        reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "Aberdeen"})]),
        reply("Taxi from Bin Mahmoud... There's also a free bus from a different station. Want the details?"),
        reply(tool_calls=[tool_call("get_bus_details", {"start": "Katara", "place": "Aberdeen"})]),
        reply("Take Metrolink bus M103 from Msheireb."),
    ])
    bot = MetroMate(llm=llm)
    bot.ask("How do I get to Aberdeen from Katara?")
    first_tool_result = llm.calls[1]["messages"][-1]["content"]
    assert "M103" not in first_tool_result                    # hidden at first
    bot.ask("yes please")
    bus_tool_result = llm.calls[3]["messages"][-1]["content"]
    assert "M103" in bus_tool_result                          # revealed when asked
    assert "How do I get to Aberdeen" in json.dumps(llm.calls[2]["messages"])   # it remembers the question


# ---------- memory ----------

def test_memory_keeps_only_questions_and_answers():
    llm = FakeLLM([
        reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "Mall of Qatar"})]),
        reply("Answer one."),
        reply("Answer two."),
    ])
    bot = MetroMate(llm=llm)
    bot.ask("first question")
    assert [m["role"] for turn in bot.turns for m in turn] == ["user", "assistant"]   # no tool messages kept
    bot.ask("second question")
    sent = llm.calls[2]["messages"]
    assert roles(sent) == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == "first question" and sent[2]["content"] == "Answer one."


def test_memory_is_trimmed_to_the_last_few_exchanges():
    llm = FakeLLM([reply(f"answer {i}") for i in range(12)])
    bot = MetroMate(llm=llm)
    for i in range(12):
        bot.ask(f"question {i}")
    last = llm.calls[-1]["messages"]
    assert len(last) == 1 + 2 * KEEP_TURNS + 1                # system + remembered pairs + new question
    assert "question 0" not in json.dumps(last)               # old exchanges are forgotten
    assert last[-1]["content"] == "question 11"


# ---------- things going wrong ----------

def test_the_loop_cannot_run_forever():
    llm = FakeLLM([reply(tool_calls=[tool_call("plan_route", {"start": "Katara", "end": "Al Sadd"}, f"c{i}")])
                   for i in range(MAX_TOOL_ROUNDS + 3)])
    bot = MetroMate(llm=llm)
    answer = bot.ask("loop please")
    assert "too many steps" in answer
    assert len(llm.calls) == MAX_TOOL_ROUNDS
    assert bot.turns == []                                    # a failed question isn't remembered


def test_empty_answers_get_a_friendly_message():
    bot = MetroMate(llm=FakeLLM([reply("   ")]))
    assert "couldn't put an answer together" in bot.ask("hi")
    assert bot.turns == []


def test_errors_become_friendly_messages_never_crashes():
    assert MetroMate(llm=FakeLLM([LLMError("I can't reach Groq.")])).ask("hi") == "I can't reach Groq."
    assert "30 seconds" in MetroMate(llm=FakeLLM([RateLimited(30)])).ask("hi")
    assert "usage limit" in MetroMate(llm=FakeLLM([RateLimited(None)])).ask("hi")
    assert "usage limit" in MetroMate(llm=FakeLLM([RateLimited(3600)])).ask("hi")


def test_rate_limit_message_and_header_parsing():
    assert "12 seconds" in rate_limit_message(12)
    assert parse_retry_after("12") == 12
    assert parse_retry_after("12.4") == 13
    assert parse_retry_after(None) is None
    assert parse_retry_after("soon") is None


# ---------- not repeating a route the visitor just saw ----------

def tool_result(llm, call_number):
    """The tool message the model was given in its call number `call_number` (0 = first)."""
    return json.loads(llm.calls[call_number]["messages"][-1]["content"])


def test_a_route_is_not_repeated_for_the_same_start_and_station():
    llm = FakeLLM([reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "QU"})]), reply("Which building?"),
                   reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "College of Engineering"})]), reply("A taxi.")])
    bot = MetroMate(llm=llm)
    bot.ask("Katara to QU")
    bot.ask("engineering")
    first, second = tool_result(llm, 1), tool_result(llm, 3)
    assert first["route"]["steps"] and "route_already_given" not in first
    assert second["route"] == {"already_given": True} and second["route_already_given"] is True
    assert second["last_stretch"]["advice"] == "taxi"          # everything else is still there


def test_a_different_start_or_station_gets_its_route():
    llm = FakeLLM([reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "QU"})]), reply("ok"),
                   reply(tool_calls=[tool_call("plan_trip", {"start": "Al Sadd", "place": "QU Engineering"})]), reply("ok"),
                   reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "Mall of Qatar"})]), reply("ok")])
    bot = MetroMate(llm=llm)
    for question in ("a", "b", "c"):
        bot.ask(question)
    assert "route_already_given" not in tool_result(llm, 3)    # another start station
    assert "route_already_given" not in tool_result(llm, 5)    # another closest station


def test_a_route_that_was_never_delivered_is_not_marked_as_shown():
    llm = FakeLLM([reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "QU"})]), RateLimited(None),
                   reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "QU"})]), reply("ok")])
    bot = MetroMate(llm=llm)
    assert "usage limit" in bot.ask("first try fails")
    bot.ask("second try")
    assert "route_already_given" not in tool_result(llm, 3)    # the visitor never saw the first route


def test_a_route_is_shown_again_once_it_has_scrolled_out_of_memory():
    script = [reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "QU"})]), reply("ok")]
    for _ in range(KEEP_TURNS):
        script.append(reply("filler"))
    script += [reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "QU Engineering"})]), reply("ok")]
    llm = FakeLLM(script)
    bot = MetroMate(llm=llm)
    bot.ask("first")
    for i in range(KEEP_TURNS):
        bot.ask(f"filler {i}")
    bot.ask("again")
    assert "route_already_given" not in tool_result(llm, len(llm.calls) - 1)


def test_other_tools_are_never_changed():
    llm = FakeLLM([reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "Aberdeen"})]), reply("ok"),
                   reply(tool_calls=[tool_call("get_bus_details", {"start": "Katara", "place": "Aberdeen"})]), reply("ok")])
    bot = MetroMate(llm=llm)
    bot.ask("a")
    bot.ask("yes")
    assert tool_result(llm, 3)["buses"][0]["route_to_boarding_station"]["steps"]


# ---------- call_llm: fallback model, dropped settings, retries ----------

class FakeSend:
    """Replaces chatbot._send. Each step is either a reply to return or an error to raise."""
    def __init__(self, steps):
        self.steps = list(steps)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def with_fake_send(fake, check):
    saved = chatbot._send
    chatbot._send = fake
    try:
        return check()
    finally:
        chatbot._send = saved


def test_call_llm_sends_the_model_tools_and_settings():
    fake = FakeSend([reply("ok")])
    with_fake_send(fake, lambda: call_llm([{"role": "user", "content": "hi"}], TOOLS))
    payload = fake.payloads[0]
    assert payload["model"] == MODEL and payload["tools"] == TOOLS
    assert payload["tool_choice"] == "auto"
    assert "reasoning_effort" in payload and "max_completion_tokens" in payload


def test_call_llm_switches_to_the_fallback_model_when_rate_limited():
    fake = FakeSend([RateLimited(10), reply("from the backup")])
    result = with_fake_send(fake, lambda: call_llm([], TOOLS))
    assert result["choices"][0]["message"]["content"] == "from the backup"
    assert [p["model"] for p in fake.payloads] == [MODEL, FALLBACK_MODEL]


def test_call_llm_gives_up_when_both_models_are_rate_limited():
    fake = FakeSend([RateLimited(10), RateLimited(25)])
    try:
        with_fake_send(fake, lambda: call_llm([], TOOLS))
        assert False, "should have raised RateLimited"
    except RateLimited as error:
        assert error.retry_after == 25


def test_call_llm_drops_a_setting_groq_does_not_support():
    fake = FakeSend([BadRequest("`reasoning_effort` is not supported for this model"), reply("ok")])
    with_fake_send(fake, lambda: call_llm([], TOOLS))
    assert "reasoning_effort" in fake.payloads[0]
    assert "reasoning_effort" not in fake.payloads[1]          # dropped on the retry
    assert "max_completion_tokens" in fake.payloads[1]         # the other setting stays


def test_call_llm_retries_a_broken_tool_call():
    fake = FakeSend([BadRequest("tool_use_failed"), BadRequest("tool_use_failed"), reply("fine")])
    result = with_fake_send(fake, lambda: call_llm([], TOOLS))
    assert result["choices"][0]["message"]["content"] == "fine"
    assert len(fake.payloads) == 3


def test_call_llm_stops_retrying_broken_tool_calls():
    fake = FakeSend([BadRequest("tool_use_failed")] * 3)
    try:
        with_fake_send(fake, lambda: call_llm([], TOOLS))
        assert False, "should have raised LLMError"
    except LLMError as error:
        assert "rephras" in str(error)
    assert len(fake.payloads) == 3                             # first try + 2 retries


def test_call_llm_turns_other_bad_requests_into_a_friendly_error():
    fake = FakeSend([BadRequest("something else entirely")])
    try:
        with_fake_send(fake, lambda: call_llm([], TOOLS))
        assert False, "should have raised LLMError"
    except LLMError as error:
        assert "something else entirely" not in str(error)     # raw details aren't shown


# ---------- _send against a fake Groq server ----------

class FakeGroq:
    """A tiny local web server. `script` is a list of (status, headers, body) answers,
    used one per request. It records what it received."""
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append({"path": self.path, "headers": dict(self.headers),
                                       "body": json.loads(self.rfile.read(length))})
                status, headers, body = outer.script.pop(0)
                data = json.dumps(body).encode() if not isinstance(body, bytes) else body
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/openai/v1/chat/completions"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def with_fake_groq(script, check, key="test-key"):
    """Point chatbot at a fake server (and a fake key), run `check(fake)`, then restore."""
    fake = FakeGroq(script)
    saved_url, saved_key = chatbot.API_URL, os.environ.get("GROQ_API_KEY")
    chatbot.API_URL = fake.url
    if key is None:
        os.environ.pop("GROQ_API_KEY", None)
    else:
        os.environ["GROQ_API_KEY"] = key
    try:
        return check(fake)
    finally:
        chatbot.API_URL = saved_url
        if saved_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = saved_key
        fake.close()


def raises(error_type, function):
    try:
        function()
    except error_type as error:
        return error
    raise AssertionError(f"expected {error_type.__name__}")


def test_send_sends_the_key_the_name_and_the_message():
    def check(fake):
        result = chatbot._send({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        assert result["choices"][0]["message"]["content"] == "hello"
        request = fake.requests[0]
        assert request["path"] == "/openai/v1/chat/completions"
        assert request["headers"]["Authorization"] == "Bearer test-key"
        assert request["headers"]["User-Agent"] == "MetroMate/0.1"
        assert request["body"]["messages"][0]["content"] == "hi"
    with_fake_groq([(200, {}, reply("hello"))], check)


def test_send_turns_http_errors_into_the_right_exceptions():
    def check(fake):
        error = raises(RateLimited, lambda: chatbot._send({}))
        assert error.retry_after == 12
        assert "key" in str(raises(LLMError, lambda: chatbot._send({}))).lower()      # 401
        assert "key" in str(raises(LLMError, lambda: chatbot._send({}))).lower()      # 403
        assert "reasoning_effort" in str(raises(BadRequest, lambda: chatbot._send({})))
        assert "500" in str(raises(LLMError, lambda: chatbot._send({})))
        assert "read" in str(raises(LLMError, lambda: chatbot._send({})))             # not JSON
    with_fake_groq([(429, {"retry-after": "12"}, {"error": {}}),
                    (401, {}, {"error": {}}),
                    (403, {}, {"error": {}}),
                    (400, {}, {"error": {"message": "`reasoning_effort` unsupported"}}),
                    (500, {}, {"error": {}}),
                    (200, {}, b"this is not json")], check)


def test_send_without_a_key_explains_what_to_do():
    def check(fake):
        assert ".env" in str(raises(LLMError, lambda: chatbot._send({})))
        assert fake.requests == []                              # nothing was sent
    with_fake_groq([], check, key=None)


def test_send_when_groq_cannot_be_reached():
    saved_url, saved_key = chatbot.API_URL, os.environ.get("GROQ_API_KEY")
    chatbot.API_URL = "http://127.0.0.1:9/nothing-listens-here"
    os.environ["GROQ_API_KEY"] = "test-key"
    try:
        assert "internet" in str(raises(LLMError, lambda: chatbot._send({})))
    finally:
        chatbot.API_URL = saved_url
        if saved_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = saved_key


def test_a_whole_conversation_through_the_real_connection_code():
    # Server script: the model asks for plan_trip, then writes its final answer.
    script = [(200, {}, reply(tool_calls=[tool_call("plan_trip", {"start": "Katara", "place": "Aberdeen"})])),
              (200, {}, reply("Take the Red Line to Msheireb, then the Gold Line to Bin Mahmoud."))]

    def check(fake):
        answer = MetroMate().ask("How do I get to Aberdeen from Katara?")
        assert answer.startswith("Take the Red Line")
        first, second = fake.requests[0]["body"], fake.requests[1]["body"]
        assert first["model"] == MODEL and len(first["tools"]) == len(TOOLS)
        assert roles(first["messages"]) == ["system", "user"]
        assert roles(second["messages"]) == ["system", "user", "assistant", "tool"]
        tool_result = json.loads(second["messages"][3]["content"])
        assert tool_result["place"] == "University of Aberdeen" and tool_result["offer"]["type"] == "bus_other_station"
    with_fake_groq(script, check)


def test_rate_limits_through_the_real_connection_code():
    # The main model is rate-limited, so the backup model answers.
    script = [(429, {"retry-after": "5"}, {"error": {}}), (200, {}, reply("Answer from the backup model."))]

    def check(fake):
        assert MetroMate().ask("hi") == "Answer from the backup model."
        assert [r["body"]["model"] for r in fake.requests] == [MODEL, FALLBACK_MODEL]
    with_fake_groq(script, check)


# ---------- the instructions ----------

def test_the_instructions_have_the_key_rules_and_stay_short():
    for phrase in ["MetroMate", "offer.question", "needs_building", "tram_notice", "tram_time_unknown",
                   "get_bus_details", "get_metro_taxi_option", "park_and_ride", "qu_shuttle",
                   "Qatar Rail", "Karwa", "never guess distances", "unofficial",
                   "requested_building_not_listed", "ALL the names", "own knowledge",
                   "route_already_given", "get_walk_distance", "bulleted list",
                   "walk_to_place_after_getting_off_m", "offer is null"]:
        assert phrase in SYSTEM_PROMPT, f"instructions are missing: {phrase}"
    assert len(SYSTEM_PROMPT) < 3200                            # about 800 tokens at most


def test_greeting_names_the_bot_and_the_limits():
    assert "MetroMate" in GREETING and "unofficial" in GREETING and "estimates" in GREETING


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
        except Exception as error:                         # a crash inside the code being tested
            failed += 1
            print(f"ERROR {name}  {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed.")
