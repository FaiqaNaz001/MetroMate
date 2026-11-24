"""
web_helpers.py
--------------
The logic behind the MetroMate web page that does NOT need Streamlit, so it can be tested
without a browser. app.py (the page itself) stays thin and uses these pieces:

  * clean_user_text   tidies and checks what a visitor typed
  * UsageGuard        protects the free Groq allowance (a daily total plus a per-visit cap)
  * ToolRecorder      watches which tools the chatbot used, so the page knows which buttons to show
  * quick_replies_for turns that into tap buttons ("Yes, show me the bus details")
  * safe_markdown     stops "$" from being shown as maths
"""

import base64
import datetime
import html
import json
import re
import threading
from pathlib import Path

from chat_tools import run_tool

# ----- SETTINGS -----------------------------------------------------------
MAX_CHARS = 300          # longest message a visitor may send
SESSION_LIMIT = 20       # questions per visit (a refresh starts a new visit)
DAILY_LIMIT = 250        # questions per day across ALL visitors; keep below your Groq allowance

# Questions shown as tap buttons in the sidebar, so a visitor can try the bot in one click.
EXAMPLES = [
    "How do I get to Mall of Qatar from Katara?",
    "How do I get to Aberdeen from Katara?",
    "How do I get to Place Vendome from Al Sadd?",
    "Katara to QU",
]

SESSION_MESSAGE = ("You've reached the question limit for this visit. Please come back a "
                   "little later, or press New chat to start again.")
DAILY_MESSAGE = ("MetroMate has used up its free allowance for today. Please try again tomorrow.")
# --------------------------------------------------------------------------

# Which tap buttons to show for each kind of offer the chatbot can make.
# Each button is (label on the button, message that gets sent when it is tapped).
BUS_BUTTON = ("Yes, show me the bus details", "Yes, please give me the bus details.")
TAXI_BUTTON = ("Yes, show me the taxi option", "Yes, please tell me about the taxi from the metro station.")
OFFER_BUTTONS = {
    "bus_same_station": [BUS_BUTTON],
    "bus_other_station": [BUS_BUTTON],
    "tram_taxi": [TAXI_BUTTON],
    "tram_alternatives": [("Taxi option", TAXI_BUTTON[1]), ("Bus option", BUS_BUTTON[1])],
}


# ----- COLORS (the design) -------------------------------------------------
PALETTE = {
    "page_bg": "#2d0606",        # behind the background picture
    "bot_bubble": "#4a1b1b",     # MetroMate's chat boxes
    "user_bubble": "#702b2b",    # the visitor's chat boxes
    "bubble_text": "#ffffff",    # text inside both
    "heading": "#f0d7c0",        # "MetroMate" and the description at the top
    "sidebar_bg": "#282828",
    "sidebar_text": "#f0d7c0",
    "accent": "#e0a47a",         # button hover and highlights
}
BACKGROUND_FILE = Path(__file__).parent / "assets" / "background.jpg"
HERO_DESCRIPTION = ("An unofficial student project. Routes and times are estimates, so please "
                    "check with Qatar Rail before you travel.")
# --------------------------------------------------------------------------

URL_PATTERN = re.compile(r"(https?://[^\s<]+?)([.,;:!?)]*)(?=\s|<|$)")


def load_background(path=BACKGROUND_FILE):
    """The background picture as a 'data:' address that CSS can use, or "" if it is missing
    (the page then simply falls back to the plain page color)."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


BULLET_PATTERN = re.compile(r"^\s*(?:[-\u2022]|\*(?!\*))\s+(.*)$")


def format_text(text):
    """Make a chat message safe to put inside the page's HTML:
    escape anything that looks like HTML, turn web addresses into links, **bold** into bold,
    lines starting with "- " or a bullet into a list, new lines into line breaks, and $ into an
    entity so it can't be shown as maths."""
    text = html.escape(text)
    text = URL_PATTERN.sub(
        lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener noreferrer">{m.group(1)}</a>{m.group(2)}',
        text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text).replace("$", "&#36;")

    blocks = []                                  # each block: ("text", [lines]) or ("list", [items])
    for line in text.split("\n"):
        bullet = BULLET_PATTERN.match(line)
        kind = "list" if bullet else "text"
        if not blocks or blocks[-1][0] != kind:
            blocks.append((kind, []))
        blocks[-1][1].append(bullet.group(1) if bullet else line)

    parts = []
    for index, (kind, lines) in enumerate(blocks):
        if kind == "list":
            parts.append("<ul>" + "".join(f"<li>{item}</li>" for item in lines) + "</ul>")
            continue
        if index > 0:                             # text after a list: drop blank lines at its start
            while lines and not lines[0].strip():
                lines = lines[1:]
        if index < len(blocks) - 1:               # text before a list: drop blank lines at its end
            while lines and not lines[-1].strip():
                lines = lines[:-1]
        parts.append("<br>".join(lines))
    return "".join(parts)


def message_html(role, text):
    """One chat bubble. The visitor's go on the right, MetroMate's on the left."""
    side = "user" if role == "user" else "bot"
    return f'<div class="msg {side}"><div class="bubble">{format_text(text)}</div></div>'


def hero_html(name, description=HERO_DESCRIPTION):
    """The heading and description centered at the top of the chat."""
    return f'<div class="hero"><h1>{html.escape(name)}</h1><p>{html.escape(description)}</p></div>'


def build_css(background_uri=""):
    """All of the page's styling. Streamlit's own theme (in .streamlit/config.toml) sets the
    base colors; this adds the background picture, the chat bubbles and the finishing touches."""
    c = PALETTE
    picture = f'url("{background_uri}") center/cover fixed' if background_uri else c["page_bg"]
    rules = [
        f'.stApp{{background:{picture};background-color:{c["page_bg"]}}}',
        '[data-testid="stHeader"]{background:transparent}',
        f'[data-testid="stSidebar"]{{background:{c["sidebar_bg"]}}}',
        f'[data-testid="stSidebar"] *{{text-align:center;color:{c["sidebar_text"]}}}',
        '.block-container{max-width:1000px;padding-top:2.5rem}',
        '.hero{text-align:center;margin:0 0 28px}',
        f'.hero h1{{color:{c["heading"]};font-size:2.6rem;margin:0 0 6px;padding:0;text-shadow:0 2px 10px rgba(0,0,0,.6)}}',
        f'.hero p{{color:{c["heading"]};font-size:1.05rem;max-width:720px;margin:0 auto;line-height:1.5;text-shadow:0 1px 8px rgba(0,0,0,.6)}}',
        '.msg{display:flex;margin:14px 0}',
        '.msg.user{justify-content:flex-end}',
        f'.bubble{{max-width:72%;padding:16px 24px;border-radius:28px;color:{c["bubble_text"]};line-height:1.55;font-size:1rem;word-break:break-word}}',
        f'.msg.bot .bubble{{background:{c["bot_bubble"]}}}',
        f'.msg.user .bubble{{background:{c["user_bubble"]}}}',
        f'.bubble a{{color:{c["heading"]};text-decoration:underline}}',
        '.bubble ul{margin:8px 0;padding-left:22px}',
        '.bubble li{margin:3px 0}',
        f'.stButton button{{background:{c["sidebar_bg"]};border:1px solid rgba(240,215,192,.35);border-radius:14px}}',
        f'.stButton button:hover{{border-color:{c["accent"]}}}',
        f'.stButton button:hover p{{color:{c["accent"]}}}',
        '[data-testid="stBottom"]>div{background:transparent}',
        f'[data-testid="stChatInput"]>div{{background:{c["sidebar_bg"]};border-radius:28px;border:1px solid rgba(240,215,192,.25)}}',
        '[data-testid="stChatInput"] textarea{color:#ffffff}',
        '@media (max-width:760px){.bubble{max-width:92%}.hero h1{font-size:2rem}}',
    ]
    return "\n".join(rules)


def clean_user_text(text, max_chars=MAX_CHARS):
    """Return (clean_text, None) for a usable message, or (None, problem) when it isn't:
    empty, or longer than max_chars. The problem is a friendly sentence to show the visitor."""
    text = " ".join((text or "").split())
    if not text:
        return None, "Please type a question."
    if len(text) > max_chars:
        return None, f"Please keep your message under {max_chars} characters."
    return text, None


def safe_markdown(text):
    """The page shows text as markdown, where $...$ means maths. Escape the dollar signs."""
    return text.replace("$", "\\$")


class ToolRecorder:
    """Wraps run_tool and remembers what the chatbot just did, for the page's buttons.
    Call reset() before each question. After the answer, `offer` holds the offer type from
    plan_trip (or None) and `buildings` holds the Qatar University buildings (or None)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.offer = None
        self.buildings = None

    def __call__(self, name, arguments):
        result = run_tool(name, arguments)
        if name == "plan_trip":
            try:
                data = json.loads(result)
            except ValueError:
                return result
            offer = data.get("offer")
            self.offer = offer["type"] if offer else None
            self.buildings = data.get("buildings") if data.get("needs_building") else None
        return result


def quick_replies_for(recorder):
    """The tap buttons to show under the chatbot's latest answer, as (label, message) pairs."""
    return list(OFFER_BUTTONS.get(recorder.offer, []))


class UsageGuard:
    """Counts questions so strangers can't use up the free Groq allowance.
    One guard is shared by every visitor (the page keeps it with st.cache_resource)."""

    def __init__(self, daily_limit=DAILY_LIMIT, session_limit=SESSION_LIMIT,
                 today=datetime.date.today):
        self.daily_limit = daily_limit
        self.session_limit = session_limit
        self._today = today
        self._day = today()
        self._count = 0
        self._lock = threading.Lock()      # several visitors can arrive at once

    def _new_day_check(self):
        if self._today() != self._day:
            self._day = self._today()
            self._count = 0

    def check(self, session_questions):
        """None if a question is allowed, otherwise the message to show instead."""
        with self._lock:
            self._new_day_check()
            if self._count >= self.daily_limit:
                return DAILY_MESSAGE
            if session_questions >= self.session_limit:
                return SESSION_MESSAGE
            return None

    def record(self):
        """Count one answered question."""
        with self._lock:
            self._new_day_check()
            self._count += 1
