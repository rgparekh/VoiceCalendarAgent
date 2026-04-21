# Google Calendar Voice Agent — Streamlit UI

import os
import logging
from datetime import datetime, timezone
from typing import Optional

import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from google_calendar_voice_agent import (
    process_calendar_request,
    get_calendar_events,
    get_tasks,
)
import voice_input as vi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* ── Chat bubbles ────────────────────────────────────────────────── */
.bubble-wrap {
    display: flex;
    margin-bottom: 0.75rem;
}
.bubble-wrap.user  { justify-content: flex-end; }
.bubble-wrap.agent { justify-content: flex-start; }

.bubble {
    max-width: 72%;
    padding: 0.65rem 1rem;
    border-radius: 18px;
    line-height: 1.5;
    font-size: 0.95rem;
    word-wrap: break-word;
}
.bubble-wrap.user  .bubble {
    background: #1a73e8;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}
.bubble-wrap.agent .bubble {
    background: #f1f3f4;
    color: #202124;
    border-bottom-left-radius: 4px;
}
.bubble-wrap.agent.error .bubble {
    background: #fce8e6;
    color: #c5221f;
}
.bubble a { color: #1a73e8; }
.bubble-wrap.user .bubble a { color: #cfe2ff; }

.bubble-meta {
    font-size: 0.72rem;
    color: #80868b;
    margin-top: 0.25rem;
    padding: 0 0.4rem;
}
.bubble-wrap.user  .bubble-meta { text-align: right; }
.bubble-wrap.agent .bubble-meta { text-align: left; }

/* ── Input area ──────────────────────────────────────────────────── */
.input-shell {
    border: 1.5px solid #dadce0;
    border-radius: 12px;
    padding: 0.5rem 0.75rem 0.4rem;
    background: #ffffff;
    margin-top: 0.5rem;
}
.input-shell:focus-within {
    border-color: #1a73e8;
    box-shadow: 0 0 0 2px rgba(26,115,232,.15);
}
/* Remove Streamlit's default textarea border so our shell is the frame */
.input-shell textarea {
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    resize: none;
}
/* Toolbar row inside shell */
.input-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 0.3rem;
}

/* ── Recording pulse animation ───────────────────────────────────── */
@keyframes pulse-ring {
    0%   { transform: scale(1);    opacity: 0.7; }
    70%  { transform: scale(1.35); opacity: 0;   }
    100% { transform: scale(1.35); opacity: 0;   }
}
.rec-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.85rem;
    color: #c5221f;
    font-weight: 500;
}
.rec-dot {
    position: relative;
    width: 10px; height: 10px;
}
.rec-dot::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: #c5221f;
    animation: pulse-ring 1.4s ease-out infinite;
}
.rec-dot::after {
    content: "";
    position: absolute;
    inset: 1px;
    border-radius: 50%;
    background: #ea4335;
}

/* ── Upcoming event cards ────────────────────────────────────────── */
.event-card {
    border: 1px solid #e8eaed;
    border-radius: 8px;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.5rem;
    background: #fff;
}
.event-card:hover { border-color: #1a73e8; }
.event-title { font-weight: 600; font-size: 0.9rem; color: #202124; }
.event-meta  { font-size: 0.78rem; color: #5f6368; margin-top: 0.15rem; }
</style>
"""

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_google_credentials() -> Credentials:
    """Return valid Google credentials, refreshing or re-authorising as needed."""
    creds = None
    force_consent = False

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            with open("token.json", "w") as f:
                f.write(creds.to_json())
        except RefreshError:
            force_consent = True
            creds = None
            try:
                os.remove("token.json")
            except OSError:
                pass

    if not creds or not creds.valid:
        if not os.path.exists("credentials.json"):
            st.error("❌ credentials.json not found. Download it from Google Cloud Console.")
            st.stop()
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        kwargs = {"port": 0, "access_type": "offline"}
        if force_consent:
            kwargs["prompt"] = "consent"
        creds = flow.run_local_server(**kwargs)
        with open("token.json", "w") as f:
            f.write(creds.to_json())

    return creds


# ---------------------------------------------------------------------------
# Whisper model (cached for process lifetime)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading speech recognition model…")
def _load_whisper_model():
    return vi._get_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_upcoming_events(creds, max_results: int = 5):
    try:
        service = build("calendar", "v3", credentials=creds)
        now = datetime.now(timezone.utc).isoformat()
        result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        return result.get("items", []), None
    except HttpError as e:
        return [], str(e)


def _format_event_time(event: dict) -> str:
    start_raw = event["start"].get("dateTime", event["start"].get("date", ""))
    end_raw   = event["end"].get("dateTime",   event["end"].get("date",   ""))
    try:
        if "T" in start_raw:
            s = datetime.fromisoformat(start_raw)
            e = datetime.fromisoformat(end_raw)
            return f"{s.strftime('%a, %b %d · %I:%M %p')} – {e.strftime('%I:%M %p')}"
        else:
            return datetime.fromisoformat(start_raw).strftime("%a, %b %d") + " · All day"
    except ValueError:
        return start_raw


def _notification_controls(key_prefix: str) -> Optional[dict]:
    """Compact email + popup reminder controls. Returns API-ready dict or None."""
    unit_map = {"min": 1, "hr": 60, "day": 1440}
    overrides = []
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Email reminder", key=f"{key_prefix}_email_on"):
            c1, c2 = st.columns([1, 1])
            amt  = c1.number_input("", min_value=1, value=1,   key=f"{key_prefix}_ea", label_visibility="collapsed")
            unit = c2.selectbox("",  ["day", "hr", "min"],     key=f"{key_prefix}_eu", label_visibility="collapsed")
            overrides.append({"method": "email",  "minutes": int(amt * unit_map[unit])})
    with col2:
        if st.checkbox("Pop-up reminder", key=f"{key_prefix}_popup_on"):
            c1, c2 = st.columns([1, 1])
            amt  = c1.number_input("", min_value=1, value=30,  key=f"{key_prefix}_pa", label_visibility="collapsed")
            unit = c2.selectbox("",  ["min", "hr", "day"],     key=f"{key_prefix}_pu", label_visibility="collapsed")
            overrides.append({"method": "popup",  "minutes": int(amt * unit_map[unit])})
    return {"useDefault": False, "overrides": overrides} if overrides else None


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------

def _render_chat():
    """Render all messages in st.session_state.messages as chat bubbles."""
    for msg in st.session_state.messages:
        role = msg["role"]          # "user" | "agent" | "error"
        text = msg["text"]
        ts   = msg.get("ts", "")
        link = msg.get("link", "")

        bubble_class = "agent error" if role == "error" else role
        link_html = f'<br><a href="{link}" target="_blank">📅 View in Calendar</a>' if link else ""

        st.markdown(
            f"""
            <div class="bubble-wrap {bubble_class}">
              <div>
                <div class="bubble">{text}{link_html}</div>
                <div class="bubble-meta">{ts}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _append_user(text: str):
    st.session_state.messages.append({
        "role": "user",
        "text": text,
        "ts": datetime.now().strftime("%I:%M %p"),
    })


def _append_agent(text: str, link: str = ""):
    st.session_state.messages.append({
        "role": "agent",
        "text": text,
        "ts": datetime.now().strftime("%I:%M %p"),
        "link": link,
    })


def _append_error(text: str):
    st.session_state.messages.append({
        "role": "error",
        "text": text,
        "ts": datetime.now().strftime("%I:%M %p"),
    })


# ---------------------------------------------------------------------------
# Submit handler
# ---------------------------------------------------------------------------

def _handle_submit(creds, text: str, reminders: Optional[dict]):
    """Process a user command and append both sides to the chat history."""
    text = text.strip()
    if not text:
        return
    _append_user(text)
    with st.spinner("Thinking…"):
        try:
            result = process_calendar_request(
                creds, "primary", text, reminders_override=reminders
            )
            if result and result.success:
                _append_agent(result.message, link=result.calendar_link or "")
            else:
                msg = result.message if result else "I couldn't process that request. Please try rephrasing."
                _append_error(msg)
        except Exception as exc:
            _append_error(f"Something went wrong: {exc}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def show_chat_page(creds):
    """Main chat interface."""
    st.markdown("## 💬 Calendar Assistant")
    st.caption("Describe anything — create, modify, search, or delete meetings, events, and tasks.")

    # ── Chat history ────────────────────────────────────────────────
    if not st.session_state.messages:
        owner = st.session_state.get("calendar_owner_name", "")
        greeting = f"Hi{' ' + owner.split()[0] if owner else ''}! I'm your calendar assistant. Tell me what you'd like to do — for example:\n\n• *Schedule a sync with Alice (alice@co.com) Friday at 2 PM*\n• *Move my dentist appointment to Thursday at 10 AM*\n• *Add a task to submit the Q1 report by Friday*\n• *Delete the team standup tomorrow*"
        _append_agent(greeting)

    _render_chat()

    # ── Reminders expander ──────────────────────────────────────────
    with st.expander("🔔 Set reminders for this command"):
        st.caption("Applies to meetings and events. Not available for tasks.")
        reminders = _notification_controls("chat")
    # Store so the submit handler can read them without passing through buttons
    st.session_state["_reminders"] = reminders

    # ── Input shell ─────────────────────────────────────────────────
    st.markdown('<div class="input-shell">', unsafe_allow_html=True)

    input_key = "chat_input_text"
    if input_key not in st.session_state:
        st.session_state[input_key] = ""

    user_text = st.text_area(
        label="chat_input",
        label_visibility="collapsed",
        placeholder="Type a command or click 🎤 to speak…",
        value=st.session_state[input_key],
        height=80,
        key="_chat_textarea",
    )

    # ── Toolbar: clear · recording status · mic/stop · send ─────────
    rec_key   = "chat_recording"
    trans_key = "chat_transcript"
    if rec_key   not in st.session_state: st.session_state[rec_key]   = False
    if trans_key not in st.session_state: st.session_state[trans_key] = ""

    is_recording = st.session_state[rec_key]

    col_clear, col_recind, col_mic, col_send = st.columns([1, 3, 1, 1])

    with col_clear:
        if st.button("✕", help="Clear input", key="btn_clear", use_container_width=True):
            st.session_state[input_key]  = ""
            st.session_state[trans_key]  = ""
            st.rerun()

    with col_recind:
        if is_recording:
            st.markdown(
                '<span class="rec-indicator">'
                '<span class="rec-dot"></span>Recording…'
                '</span>',
                unsafe_allow_html=True,
            )

    with col_mic:
        if not is_recording:
            if st.button("🎤", help="Start voice input", key="btn_mic", use_container_width=True):
                vi.start_recording()
                st.session_state[rec_key]   = True
                st.session_state[trans_key] = ""
                st.rerun()
        else:
            if st.button("⏹", help="Stop and transcribe", key="btn_stop", use_container_width=True):
                audio, sr = vi.stop_recording()
                st.session_state[rec_key] = False
                with st.spinner("Transcribing…"):
                    transcript = vi.transcribe_audio(audio, sr)
                st.session_state[trans_key] = transcript
                st.session_state[input_key] = transcript
                st.rerun()

    with col_send:
        send_disabled = is_recording
        if st.button("➤", help="Send", key="btn_send",
                     type="primary", use_container_width=True,
                     disabled=send_disabled):
            final_text = user_text.strip()
            if final_text:
                _handle_submit(creds, final_text, st.session_state.get("_reminders"))
                st.session_state[input_key]  = ""
                st.session_state[trans_key]  = ""
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Sync textarea edits back into session_state so the value persists
    if user_text != st.session_state[input_key]:
        st.session_state[input_key] = user_text


def show_home_page(creds):
    """Home page showing upcoming events."""
    owner = st.session_state.get("calendar_owner_name", "")
    greeting = f"Welcome{', ' + owner if owner else ''}!"
    st.markdown(f"## 🏠 {greeting}")

    today = datetime.now().strftime("%A, %B %d, %Y")
    st.caption(f"Today is {today}")
    st.markdown("---")
    st.markdown("### 🗓️ Upcoming Events")

    events, error = _get_upcoming_events(creds, max_results=7)
    if error:
        st.error(f"Could not load upcoming events: {error}")
    elif not events:
        st.info("📭 No upcoming events found.")
    else:
        for event in events:
            title = event.get("summary", "Untitled")
            link  = event.get("htmlLink", "")
            loc   = event.get("location", "")
            has_a = bool(event.get("attendees"))
            label = "Meeting" if has_a else "Event"
            time_str = _format_event_time(event)

            title_html = f'<a href="{link}" target="_blank">{title}</a>' if link else title
            meta_parts = [f"{'👥' if has_a else '📅'} {label}", f"🕐 {time_str}"]
            if loc:
                meta_parts.append(f"📍 {loc}")

            st.markdown(
                f'<div class="event-card">'
                f'<div class="event-title">{title_html}</div>'
                f'<div class="event-meta">{" · ".join(meta_parts)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def show_settings_page():
    st.markdown("## ⚙️ Settings")

    st.markdown("### 🔑 Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Files**")
        st.success("✅ credentials.json") if os.path.exists("credentials.json") else st.error("❌ credentials.json missing")
        st.success("✅ token.json") if os.path.exists("token.json") else st.info("ℹ️ token.json — created on first use")
    with col2:
        st.markdown("**Environment**")
        st.success("✅ GOOGLE_API_KEY set") if os.environ.get("GOOGLE_API_KEY") else st.error("❌ GOOGLE_API_KEY not set")
        model_name = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash (default)")
        st.info(f"🤖 Model: {model_name}")

    st.markdown("### 🧹 Clear Data")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Authentication Token", use_container_width=True):
            if os.path.exists("token.json"):
                os.remove("token.json")
                st.success("Token cleared. You'll re-authenticate on next use.")
            else:
                st.info("No token to clear.")
    with col2:
        if st.button("💬 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Calendar Assistant",
        page_icon="📅",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # Pre-load Whisper model
    _load_whisper_model()

    # ── Session state init ───────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "page" not in st.session_state:
        st.session_state.page = "💬 Chat"

    # ── Sidebar ──────────────────────────────────────────────────────
    st.sidebar.title("📅 Calendar Assistant")

    pages = ["💬 Chat", "🏠 Upcoming Events", "⚙️ Settings"]
    page = st.sidebar.selectbox(
        "Navigate",
        pages,
        index=pages.index(st.session_state.page),
    )
    if page != st.session_state.page:
        st.session_state.page = page
        st.rerun()

    # ── API key gate ─────────────────────────────────────────────────
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY environment variable not set.")
        api_key = st.text_input("Enter your Google API Key:", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.rerun()
        st.stop()

    # ── Google auth ──────────────────────────────────────────────────
    try:
        creds = get_google_credentials()
        st.sidebar.success("✅ Connected to Google Calendar")
    except Exception as e:
        st.error(f"❌ Authentication failed: {e}")
        st.stop()

    # Cache calendar owner name
    if "calendar_owner_name" not in st.session_state:
        try:
            service = build("calendar", "v3", credentials=creds)
            info = service.calendars().get(calendarId="primary").execute()
            st.session_state.calendar_owner_name = info.get("summary", "")
        except Exception:
            st.session_state.calendar_owner_name = ""

    # ── Render page ──────────────────────────────────────────────────
    if page == "💬 Chat":
        show_chat_page(creds)
    elif page == "🏠 Upcoming Events":
        show_home_page(creds)
    elif page == "⚙️ Settings":
        show_settings_page()


if __name__ == "__main__":
    main()
