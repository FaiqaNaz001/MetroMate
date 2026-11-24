"""
test_web_helpers.py
-------------------
Tests for the MetroMate web page. Run with:
    python test_web_helpers.py

Part 1 tests web_helpers.py directly (no Streamlit needed).

Part 2 runs the whole of app.py against FakeStreamlit, a small stand-in that records what the
page draws and lets the tests type messages and tap buttons. It copies how real Streamlit
behaves (the script reruns from the top, and a tapped button's callback runs BEFORE the rerun),
so it catches mistakes in the page's logic. It can't tell you how the page LOOKS or whether
real Streamlit accepts every call; for that, run the page and look at it.
"""

import contextlib
import html
import json
import os
import re
import runpy
import sys
import threading
import types
from pathlib import Path

import chatbot
import chat_tools
from chat_tools import plan_trip
from landmarks import LANDMARKS
from test_chatbot import FakeSend, reply, tool_call
from web_helpers import (
    clean_user_text, safe_markdown, ToolRecorder, quick_replies_for, UsageGuard,
    OFFER_BUTTONS, EXAMPLES, MAX_CHARS, SESSION_LIMIT, DAILY_LIMIT, SESSION_MESSAGE, DAILY_MESSAGE,
    PALETTE, BACKGROUND_FILE, format_text, message_html, hero_html, build_css, load_background,
)

APP_PATH = str(Path(__file__).parent / "app.py")


# =============================== PART 1: the helpers ===============================

def test_clean_user_text():
    assert clean_user_text("  How   do I\n get there? ") == ("How do I get there?", None)
    text, problem = clean_user_text("   ")
    assert text is None and "type a question" in problem
    text, problem = clean_user_text("x" * (MAX_CHARS + 1))
    assert text is None and str(MAX_CHARS) in problem
    assert clean_user_text("x" * MAX_CHARS)[0] == "x" * MAX_CHARS      # exactly the limit is fine
    assert clean_user_text(None)[0] is None


def test_safe_markdown_escapes_dollar_signs():
    assert safe_markdown("A taxi costs $5") == "A taxi costs \\$5"
    assert safe_markdown("no money here") == "no money here"


def test_recorder_notes_the_offer_and_the_qu_buildings():
    recorder = ToolRecorder()
    recorder("plan_trip", json.dumps({"start": "Katara", "place": "Aberdeen"}))
    assert recorder.offer == "bus_other_station" and recorder.buildings is None
    recorder.reset()
    assert recorder.offer is None
    recorder("plan_trip", json.dumps({"start": "Katara", "place": "QU"}))
    assert recorder.offer is None and len(recorder.buildings) == 15
    recorder.reset()
    recorder("get_bus_details", json.dumps({"start": "Katara", "place": "Aberdeen"}))
    assert recorder.offer is None                      # only plan_trip sets an offer


def test_recorder_still_returns_the_tool_result_unchanged():
    result = ToolRecorder()("plan_trip", json.dumps({"start": "Katara", "place": "Mall of Qatar"}))
    assert json.loads(result)["place"] == "Mall of Qatar"
    assert json.loads(ToolRecorder()("nope", "{}"))["error"] == "unknown_tool"


def test_quick_replies():
    recorder = ToolRecorder()
    assert quick_replies_for(recorder) == []
    recorder.offer = "bus_same_station"
    assert [label for label, _ in quick_replies_for(recorder)] == ["Yes, show me the bus details"]
    recorder.offer = "tram_alternatives"
    assert [label for label, _ in quick_replies_for(recorder)] == ["Taxi option", "Bus option"]
    recorder.offer, recorder.buildings = None, ["College of Law", "Research Complex"]
    assert quick_replies_for(recorder) == []         # the buildings are listed in the chat, not as buttons


def test_every_offer_the_chatbot_can_make_has_buttons():
    seen = set()
    for place in LANDMARKS:
        offer = plan_trip("Katara", LANDMARKS[place]["name"]).get("offer")
        if offer:
            seen.add(offer["type"])
    assert seen, "expected some offers"
    assert seen <= set(OFFER_BUTTONS), f"offer types without buttons: {seen - set(OFFER_BUTTONS)}"


def test_usage_guard_limits():
    guard = UsageGuard(daily_limit=3, session_limit=2)
    assert guard.check(0) is None
    assert guard.check(2) == SESSION_MESSAGE             # this visitor asked enough
    for _ in range(3):
        guard.record()
    assert guard.check(0) == DAILY_MESSAGE               # everyone together asked enough


def test_usage_guard_starts_a_new_day():
    day = [1]
    guard = UsageGuard(daily_limit=1, session_limit=5, today=lambda: day[0])
    guard.record()
    assert guard.check(0) == DAILY_MESSAGE
    day[0] = 2
    assert guard.check(0) is None


def test_usage_guard_counts_correctly_with_many_visitors_at_once():
    guard = UsageGuard(daily_limit=10_000, session_limit=5)
    threads = [threading.Thread(target=lambda: [guard.record() for _ in range(100)]) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert guard._count == 2000


def luminance(hex_color):
    channels = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(first, second):
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_the_colors_are_the_agreed_ones():
    assert PALETTE["user_bubble"] == "#702b2b" and PALETTE["bot_bubble"] == "#4a1b1b"
    assert PALETTE["heading"] == "#f0d7c0" and PALETTE["sidebar_bg"] == "#282828"
    assert PALETTE["bubble_text"] == "#ffffff" and PALETTE["sidebar_text"] == "#f0d7c0"


def test_text_is_easy_to_read_on_every_background():
    for bubble in ("bot_bubble", "user_bubble"):
        assert contrast(PALETTE["bubble_text"], PALETTE[bubble]) >= 7, bubble
    assert contrast(PALETTE["sidebar_text"], PALETTE["sidebar_bg"]) >= 7


def test_css_uses_the_palette_and_the_picture():
    css = build_css("data:image/jpeg;base64,AAAA")
    for color in ("#702b2b", "#4a1b1b", "#f0d7c0", "#282828", "#ffffff"):
        assert color in css, color
    assert 'url("data:image/jpeg;base64,AAAA")' in css
    assert "url(" not in build_css("")                  # no picture: plain color, nothing broken
    assert "\n\n" not in css                           # a blank line would end the HTML block early


def test_background_picture_exists_and_is_small():
    assert BACKGROUND_FILE.exists(), "put background.jpg in the assets folder"
    assert BACKGROUND_FILE.stat().st_size < 400_000      # it is sent with every page update
    assert load_background().startswith("data:image/jpeg;base64,")
    assert load_background("no/such/file.jpg") == ""


def test_messages_are_made_safe_for_the_page():
    assert "<script>" not in format_text("<script>alert(1)</script>")
    assert "&lt;script&gt;" in format_text("<script>alert(1)</script>")
    assert format_text("line one\nline two") == "line one<br>line two"
    assert format_text("It costs $5") == "It costs &#36;5"
    assert format_text("**Bold** text") == "<strong>Bold</strong> text"
    assert "Tom &amp; Jerry" in format_text("Tom & Jerry")


def test_bulleted_lines_become_a_list():
    assert format_text("- A\n- B") == "<ul><li>A</li><li>B</li></ul>"
    assert format_text("\u2022 A\n\u2022 B") == "<ul><li>A</li><li>B</li></ul>"
    assert format_text("Pick one:\n\n- A\n- B\n\nWhich?") == "Pick one:<ul><li>A</li><li>B</li></ul>Which?"
    assert format_text("**Bold** and a * star") == "<strong>Bold</strong> and a * star"   # not a list
    assert "<ul>" not in format_text("A well-known place")                            # a dash inside a line isn't one
    assert ".bubble ul" in build_css("")


def test_web_addresses_become_links_without_the_full_stop():
    out = format_text("Schedule: https://www.qu.edu.qa/files/Bus%20Schedules_0.pdf.")
    assert out.count('<a href="https://www.qu.edu.qa/files/Bus%20Schedules_0.pdf"') == 1
    assert 'rel="noopener noreferrer"' in out and out.endswith("</a>.")
    assert "<a " not in format_text("javascript:alert(1) and ftp://x.y")     # only http(s)


def test_bubbles_and_hero():
    assert message_html("user", "hi") == '<div class="msg user"><div class="bubble">hi</div></div>'
    assert message_html("assistant", "hi") == '<div class="msg bot"><div class="bubble">hi</div></div>'
    hero = hero_html("MetroMate")
    assert "<h1>MetroMate</h1>" in hero and "unofficial" in hero


def test_settings_are_sensible():
    assert 0 < SESSION_LIMIT <= DAILY_LIMIT <= 300       # below the ~400 questions/day Groq estimate
    assert len(EXAMPLES) >= 3 and all(len(e) <= MAX_CHARS for e in EXAMPLES)


# =============================== PART 2: the page ===============================

class Box:
    """Something you can write `with ...:` around (the sidebar, a column)."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSessionState(dict):
    """Like st.session_state: items can be read and written as attributes too."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


BUBBLE = re.compile(r'<div class="msg (user|bot)"><div class="bubble">(.*)</div></div>', re.S)


def plain_text(fragment):
    """The visible text of a bubble's HTML: line breaks restored, tags removed, entities decoded."""
    return html.unescape(re.sub(r"<[^>]+>", "", fragment.replace("<br>", "\n")))


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = FakeSessionState()
        self.secrets = {}
        self.sidebar = Box()
        self.events = []             # what the page drew during the latest run
        self.typed = None            # what the chat box returns on the next run
        self.registered = {}         # button label -> (callback, args), from the latest run
        self.cache = {}
        self.role = None

    # --- drawing ---
    def set_page_config(self, **kwargs):
        pass

    def title(self, text):
        self.events.append(("title", text))

    def caption(self, text):
        self.events.append(("caption", text))

    def markdown(self, text, unsafe_allow_html=False):
        bubble = BUBBLE.fullmatch(text)
        if bubble:                                   # a chat bubble: remember who said what
            role = "user" if bubble.group(1) == "user" else "assistant"
            self.events.append(("bubble", role, plain_text(bubble.group(2)), bubble.group(2)))
        else:
            self.events.append(("markdown", self.role, text))

    def error(self, text):
        self.events.append(("error", text))

    @contextlib.contextmanager
    def chat_message(self, role):
        before, self.role = self.role, role
        try:
            yield
        finally:
            self.role = before

    @contextlib.contextmanager
    def spinner(self, text=""):
        yield

    def columns(self, count):
        return [Box() for _ in range(count)]

    # --- input ---
    def chat_input(self, placeholder=""):
        value, self.typed = self.typed, None
        return value

    def button(self, label, on_click=None, args=(), key=None):
        self.events.append(("button", label))
        self.registered[label] = (on_click, args)
        return False

    def cache_resource(self, function):
        def wrapper(*args, **kwargs):
            if function.__name__ not in self.cache:
                self.cache[function.__name__] = function(*args, **kwargs)
            return self.cache[function.__name__]
        return wrapper


class Page:
    """Runs app.py the way Streamlit does. run() = one visit/interaction."""
    def __init__(self):
        self.st = FakeStreamlit()

    def run(self, typed=None, tap=None):
        st = self.st
        if tap is not None:                          # Streamlit runs the tapped button's callback first
            callback, args = st.registered[tap]
            callback(*args)
        st.events, st.typed, st.role = [], typed, None
        saved = sys.modules.get("streamlit")
        sys.modules["streamlit"] = st
        try:
            runpy.run_path(APP_PATH, run_name="__main__")
        finally:
            if saved is None:
                sys.modules.pop("streamlit", None)
            else:
                sys.modules["streamlit"] = saved
        return st.events

    # handy views of the latest run
    def says(self, role):
        return [e[2] for e in self.st.events if e[0] == "bubble" and e[1] == role]

    def raw_bubbles(self):
        return [e[3] for e in self.st.events if e[0] == "bubble"]

    def html_blocks(self):
        return [e[2] for e in self.st.events if e[0] == "markdown"]

    def buttons(self):
        return [e[1] for e in self.st.events if e[0] == "button"]


@contextlib.contextmanager
def page_environment(steps, key="test-key"):
    """A Page with a pretend Groq (answering `steps` in order) and a fake API key."""
    fake_send = FakeSend(steps)
    saved_send, saved_key = chatbot._send, os.environ.get("GROQ_API_KEY")
    chatbot._send = fake_send
    if key is None:
        os.environ.pop("GROQ_API_KEY", None)
    else:
        os.environ["GROQ_API_KEY"] = key
    try:
        yield Page(), fake_send
    finally:
        chatbot._send = saved_send
        if saved_key is None:
            os.environ.pop("GROQ_API_KEY", None)
        else:
            os.environ["GROQ_API_KEY"] = saved_key


def plan(start, place, call_id="c1"):
    return reply(tool_calls=[tool_call("plan_trip", {"start": start, "place": place}, call_id)])


def test_page_first_load_shows_greeting_examples_and_no_quick_replies():
    with page_environment([]) as (page, _):
        page.run()
        blocks = " ".join(page.html_blocks())
        assert "<h1>MetroMate</h1>" in blocks                      # the heading at the top
        assert "#702b2b" in blocks and "data:image/jpeg;base64," in blocks   # the styling and the picture
        assert "MetroMate" in page.says("assistant")[0] and "unofficial" in page.says("assistant")[0]
        assert page.buttons() == ["New chat"] + EXAMPLES


def test_page_answers_a_typed_question_and_offers_a_button():
    with page_environment([plan("Katara", "Aberdeen"), reply("Taxi from Bin Mahmoud. Want the bus details?")]) as (page, fake):
        page.run()
        page.run(typed="How do I get to Aberdeen from Katara?")
        assert page.says("user") == ["How do I get to Aberdeen from Katara?"]
        assert page.says("assistant")[-1] == "Taxi from Bin Mahmoud. Want the bus details?"
        assert "Yes, show me the bus details" in page.buttons()
        assert page.st.session_state.questions == 1
        assert len(fake.payloads) == 2                    # one call to ask for the tool, one for the answer


def test_tapping_the_offer_button_sends_a_yes_and_shows_the_bus_details():
    steps = [plan("Katara", "Aberdeen"), reply("Taxi... want the bus details?"),
             reply(tool_calls=[tool_call("get_bus_details", {"start": "Katara", "place": "Aberdeen"}, "c2")]),
             reply("Take Metrolink bus M103 from Msheireb.")]
    with page_environment(steps) as (page, fake):
        page.run(typed="Aberdeen from Katara please")
        page.run(tap="Yes, show me the bus details")
        assert page.says("user")[-1] == "Yes, please give me the bus details."   # the page redraws the whole chat
        assert page.says("assistant")[-1] == "Take Metrolink bus M103 from Msheireb."
        tool_message = fake.payloads[3]["messages"][-1]
        assert tool_message["role"] == "tool" and "M103" in tool_message["content"]
        assert "Yes, show me the bus details" not in page.buttons()      # the offer is used up


def test_qu_buildings_are_listed_in_the_chat_not_as_buttons():
    steps = [plan("Katara", "QU"), reply("Which building?\n- College of Law\n- Research Complex"),
             plan("Katara", "College of Engineering", "c2"), reply("A taxi is best.")]
    with page_environment(steps) as (page, _):
        page.run(typed="Katara to QU")
        assert page.buttons() == ["New chat"] + EXAMPLES            # no building buttons
        assert "<ul><li>College of Law</li><li>Research Complex</li></ul>" in page.raw_bubbles()[-1]
        page.run(typed="College of Engineering")
        assert page.says("assistant")[-1] == "A taxi is best."


def test_tapping_an_example_question_asks_it_immediately():
    steps = [plan("Katara", "Mall of Qatar"), reply("Take the Red and Green lines.")]
    with page_environment(steps) as (page, _):
        page.run()
        page.run(tap=EXAMPLES[0])
        assert page.says("user") == [EXAMPLES[0]]
        assert page.says("assistant")[-1] == "Take the Red and Green lines."


def test_new_chat_clears_the_conversation_and_the_counter():
    with page_environment([reply("Hello there.")]) as (page, _):
        page.run(typed="hi")
        assert page.st.session_state.questions == 1
        page.run(tap="New chat")
        assert page.st.session_state.questions == 0
        assert len(page.st.session_state.messages) == 1
        assert page.says("user") == []


def test_the_page_remembers_earlier_messages_between_runs():
    with page_environment([reply("First answer."), reply("Second answer.")]) as (page, fake):
        page.run(typed="first question")
        page.run(typed="second question")
        assert [m["content"] for m in page.st.session_state.messages if m["role"] == "user"] == \
               ["first question", "second question"]
        sent = fake.payloads[1]["messages"]               # the second request carries the first exchange
        assert any(m.get("content") == "first question" for m in sent)


def test_empty_and_too_long_messages_never_reach_the_model():
    with page_environment([]) as (page, fake):           # no scripted replies: any model call would fail
        page.run(typed="   ")
        assert "type a question" in page.says("assistant")[-1]
        page.run(typed="x" * (MAX_CHARS + 1))
        assert str(MAX_CHARS) in page.says("assistant")[-1]
        assert fake.payloads == []
        assert page.st.session_state.questions == 0


def test_visit_limit_stops_questions_without_calling_the_model():
    with page_environment([reply("ok")]) as (page, fake):
        page.run()
        page.st.session_state.questions = SESSION_LIMIT
        page.run(typed="one more please")
        assert page.says("assistant")[-1] == SESSION_MESSAGE
        assert fake.payloads == []


def test_daily_limit_stops_questions_for_everyone():
    with page_environment([reply("ok")]) as (page, fake):
        page.run()
        guard = UsageGuard()                              # the page's shared counter, already full
        guard._count = DAILY_LIMIT
        page.st.cache["get_guard"] = guard
        page.run(typed="question")
        assert page.says("assistant")[-1] == DAILY_MESSAGE
        assert fake.payloads == []


def test_dollar_signs_in_answers_cannot_turn_into_maths():
    with page_environment([reply("A taxi may cost $20.")]) as (page, _):
        page.run(typed="price?")
        assert page.says("assistant")[-1] == "A taxi may cost $20."          # what the visitor reads
        assert "$" not in page.raw_bubbles()[-1] and "&#36;20" in page.raw_bubbles()[-1]
        assert page.st.session_state.messages[-1]["content"] == "A taxi may cost $20."   # stored unchanged


def test_html_in_messages_cannot_break_the_page():
    with page_environment([reply("<script>alert(1)</script> done")]) as (page, _):
        page.run(typed="<b>hi</b>")
        raw = " ".join(page.raw_bubbles())
        assert "<script>" not in raw and "<b>" not in raw
        assert page.says("user")[-1] == "<b>hi</b>"               # shown as typed, not as HTML


def test_missing_key_shows_a_clear_error_and_stops():
    with page_environment([], key=None) as (page, fake):
        page.run()
        errors = [e for e in page.st.events if e[0] == "error"]
        assert errors and "GROQ_API_KEY" in errors[0][1]
        assert "bot" not in page.st.session_state
        assert page.buttons() == []


def test_key_can_come_from_streamlit_secrets():
    with page_environment([reply("hi")], key=None) as (page, _):
        page.st.secrets = {"GROQ_API_KEY": "key-from-secrets"}
        page.run(typed="hello")
        assert os.environ["GROQ_API_KEY"] == "key-from-secrets"
        assert page.says("assistant")[-1] == "hi"


def test_model_errors_appear_as_friendly_messages_in_the_chat():
    with page_environment([chatbot.RateLimited(None), chatbot.RateLimited(None)]) as (page, _):
        page.run(typed="hello")
        assert "usage limit" in page.says("assistant")[-1]


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
        except Exception as error:                         # a crash in the page itself
            failed += 1
            print(f"ERROR {name}  {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed.")
