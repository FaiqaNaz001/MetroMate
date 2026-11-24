"""
app.py
------
The MetroMate web page, built with Streamlit. It only handles the display: the conversation
itself comes from chatbot.py, and the helper logic from web_helpers.py.

Run it from the terminal:
    python -m streamlit run app.py

How a page works in Streamlit: every time a visitor does something (types, taps a button),
Streamlit runs this whole script again from the top. Anything that must survive between runs
lives in st.session_state (one per visitor) or in a cached resource (shared by everyone).
"""

import os

import streamlit as st

from chatbot import MetroMate, GREETING, NAME      # importing chatbot also loads your .env file
from web_helpers import (
    ToolRecorder, UsageGuard, quick_replies_for, clean_user_text,
    message_html, hero_html, build_css, load_background,
    EXAMPLES, DAILY_LIMIT, SESSION_LIMIT,
)

st.set_page_config(page_title=NAME, page_icon="🚇", layout="wide")


@st.cache_resource
def get_guard():
    """One counter shared by ALL visitors, protecting the free Groq allowance."""
    return UsageGuard(DAILY_LIMIT, SESSION_LIMIT)


@st.cache_resource
def get_background():
    """The background picture, read once and kept in memory."""
    return load_background()


def show_message(role, text):
    """Draw one chat bubble (the visitor's on the right, MetroMate's on the left)."""
    st.markdown(message_html(role, text), unsafe_allow_html=True)


def load_api_key():
    """True if a Groq key is available: from your .env file / the environment, or (when the
    app is deployed) from Streamlit's secrets. The key is never written in the code."""
    if os.environ.get("GROQ_API_KEY"):
        return True
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        return True
    except Exception:          # no secrets file, or no such secret
        return False


def new_chat():
    """Start a fresh conversation for this visitor."""
    recorder = ToolRecorder()
    st.session_state.recorder = recorder
    st.session_state.bot = MetroMate(tool_runner=recorder)
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]
    st.session_state.quick_replies = []
    st.session_state.questions = 0


def send_later(text):
    """Called when a button is tapped: remember the message, and the page reruns and sends it."""
    st.session_state.pending = text


def show_quick_replies(replies):
    """Tap buttons under the latest answer. Many buttons (the QU buildings) go in 3 columns."""
    columns = st.columns(3) if len(replies) > 3 else None
    for index, (label, text) in enumerate(replies):
        key = f"quick_{st.session_state.questions}_{index}"
        if columns:
            with columns[index % 3]:
                st.button(label, on_click=send_later, args=(text,), key=key)
        else:
            st.button(label, on_click=send_later, args=(text,), key=key)


def answer(prompt):
    """Handle one message from the visitor: check it, ask MetroMate, show and remember both."""
    text, problem = clean_user_text(prompt)
    if text is None:                                   # empty or too long
        st.session_state.messages.append({"role": "assistant", "content": problem})
        show_message("assistant", problem)
        return

    st.session_state.messages.append({"role": "user", "content": text})
    show_message("user", text)

    guard = get_guard()
    reply = guard.check(st.session_state.questions)
    if reply is None:
        st.session_state.recorder.reset()
        with st.spinner("Thinking..."):
            reply = st.session_state.bot.ask(text)
        guard.record()
        st.session_state.questions += 1
        st.session_state.quick_replies = quick_replies_for(st.session_state.recorder)
    else:                                              # a usage limit was reached
        st.session_state.quick_replies = []
    show_message("assistant", reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})


def main():
    st.markdown(f"<style>{build_css(get_background())}</style>", unsafe_allow_html=True)
    st.markdown(hero_html(NAME), unsafe_allow_html=True)

    if not load_api_key():
        st.error("I can't find a Groq API key. Add GROQ_API_KEY to your .env file "
                 "(or to the app's secrets when it is deployed).")
        return
    if "bot" not in st.session_state:
        new_chat()

    with st.sidebar:
        st.markdown(f"### {NAME}")
        st.markdown("Ask how to get to malls and universities in Doha by Metro and the "
                    "Lusail Tram.")
        st.button("New chat", on_click=new_chat, key="new_chat")
        st.markdown("**Try asking:**")
        for index, example in enumerate(EXAMPLES):
            st.button(example, on_click=send_later, args=(example,), key=f"example_{index}")
        st.caption("Covers malls and universities (including buildings at Qatar University). "
                   "Distances were collected from Google Maps in November 2026.")

    for message in st.session_state.messages:
        show_message(message["role"], message["content"])

    prompt = None
    if "pending" in st.session_state:                  # a button was tapped
        prompt = st.session_state.pending
        del st.session_state.pending
    typed = st.chat_input("Ask about getting to a mall or university")
    prompt = prompt or typed

    if prompt:
        answer(prompt)
    show_quick_replies(st.session_state.quick_replies)


main()
