# Google Calendar Voice Agent

A conversational calendar assistant that understands **natural language and voice commands** to create, search, modify, and delete meetings, events, tasks, birthdays, and anniversaries in Google Calendar. Powered by the Google Gemini API, with a chat-style Streamlit UI and speech-to-text via OpenAI Whisper.

---

## Features

### 🎤 Voice + Text Input
Speak or type your command — both work identically. Click **Start Recording**, say what you want, click **Stop Recording**, and the assistant transcribes and executes your request. The transcription is powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper medium model) running fully offline on your machine.

### 💬 Chat Interface
All interactions happen in a single chat window. There are no separate Create / Modify / Delete pages — just describe what you want and the agent figures out the action automatically.

### 📅 Meetings
Calendar events with one or more invited attendees. Google Calendar sends invitations to each attendee.

### 🗓️ Personal Events
Calendar entries owned only by you — focus blocks, appointments, all-day events, and anything else with no external attendees.

### ✅ Tasks
To-do items managed via Google Tasks. No time slot required.

### 🎂 Birthdays & Anniversaries
Yearly recurring all-day events. Created once, they repeat every year automatically on the same date. When you specify only a month and day, the agent schedules the next upcoming occurrence. Both types include default reminders:

| Type | When |
|------|------|
| Email | 15 minutes before |
| Pop-up | 15 minutes before |

### 🔔 Notifications
Email and pop-up reminders for meetings and events, configurable via the UI or inline in your command. Tasks do not support notifications via the Google API.

---

## Examples

The agent understands intent automatically — you don't need to specify whether you're creating, modifying, or deleting; just describe what you want.

### Meetings

| Intent | Example |
|--------|---------|
| Create | `Schedule a sync with Alice (alice@co.com) and Bob (bob@co.com) on Friday at 2 PM for 1 hour` |
| Create recurring | `Set up a weekly team standup every Monday at 9 AM with the team (team@co.com) for 30 minutes` |
| Create with location | `Book a client dinner with Sarah (sarah@client.com) at Nobu on Thursday at 7 PM for 2 hours` |
| Modify time | `Move the Friday sync with Alice to 3 PM` |
| Add attendee | `Add Carol (carol@co.com) to the Monday standup` |
| Delete | `Cancel the client dinner with Sarah on Thursday` |

### Personal Events

| Intent | Example |
|--------|---------|
| Create | `Block my calendar for deep work on Monday from 9 AM to 12 PM` |
| Create | `Add a dentist appointment on May 10 at 10 AM` |
| Create all-day | `Add a personal day on Friday` |
| Modify time | `Move my dentist appointment to 11 AM` |
| Modify location | `Update my dentist appointment location to 123 Main St` |
| Delete | `Remove the focus block on Monday morning` |

### Tasks

| Intent | Example |
|--------|---------|
| Create | `Add a task to submit the Q1 report by end of this week` |
| Create | `Remind me to buy groceries` |
| Create with due date | `Add a task to review the project proposal — due Friday` |
| Modify | `Update the Q1 report task notes to include the finance team review` |
| Complete | `Mark the grocery task as completed` |
| Delete | `Delete the task to review the project proposal` |

### Birthdays

The event is created as **"&lt;Name&gt;'s Birthday"**, repeating every year.

| Example |
|---------|
| `Create Alice's birthday on June 15` |
| `Add John Smith's birthday on March 3rd` |
| `Set up a birthday for my mom on October 22` |

### Anniversaries

The event is created as **"&lt;Name(s)&gt;'s &lt;Type&gt; Anniversary"**, repeating every year. The agent infers the anniversary type from context.

| Example |
|---------|
| `Add our wedding anniversary on July 4` |
| `Create John and Jane's wedding anniversary on September 12` |
| `Add Bob's work anniversary on January 15` |
| `Create my parents' 30th anniversary on August 20` |

### Notifications

Reminders can be set via the **🔔 Set reminders** panel in the UI, or described inline in your command:

| Example |
|---------|
| `Schedule a dentist appointment on Friday at 10 AM — email me 1 day before` |
| `Add a focus block Monday 9–12 PM with a pop-up reminder 15 minutes before` |
| `Book a team sync with Alice (alice@co.com) tomorrow at 2 PM — email 1 day before and pop-up 30 minutes before` |

---

## Requirements

- Python 3.10+
- A [Google Cloud](https://console.cloud.google.com/) project with:
  - **Google Calendar API** enabled
  - **Google Tasks API** enabled
  - An **OAuth consent screen** configured
  - An **OAuth 2.0 Client ID** of type **Desktop app** — download as `credentials.json`
- A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
- A microphone (for voice input)

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/VoiceCalendarAgent.git
cd VoiceCalendarAgent
```

**2. Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

The first time you use voice input, the Whisper `medium` model weights (~460 MB) are downloaded automatically to your local cache. Subsequent runs are instant.

**4. Configure secrets**

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your-gemini-api-key
LLM_MODEL_NAME=gemini-2.5-flash
```

Load it before starting the app:

```bash
export $(cat .env | grep -v '^#' | xargs)
```

Place `credentials.json` (downloaded from Google Cloud Console) in the project root.

> `.env` and `credentials.json` are listed in `.gitignore` and will never be committed.

**5. Run the app**

```bash
streamlit run google_calendar_voice_agent_ui.py
```

A browser window opens. On first use, a Google sign-in prompt appears and `token.json` is saved locally for subsequent sessions.

---

## Using the App

### Chat page (default)
Type a command in the input box and press **➤**, or use voice:

1. Click **🎤** to start recording
2. Speak your command
3. Click **⏹** to stop — the transcript appears in the input box
4. Edit if needed, then press **➤** to send

The **✕** button clears the current input without affecting chat history. Chat history can be cleared from the **Settings** page.

### Upcoming Events page
Shows your next 7 calendar events with title, time, location, and a direct link to Google Calendar.

### Settings page
- View the status of required files and environment variables
- Clear the Google authentication token (forces re-login)
- Clear the full chat history

---

## Project Layout

| File | Role |
|------|------|
| `google_calendar_voice_agent_ui.py` | Streamlit chat UI, OAuth flow, voice recording widget |
| `google_calendar_voice_agent.py` | Agent logic: LLM classification, routing, and Google API calls |
| `voice_input.py` | Microphone recording (`sounddevice`) and Whisper transcription (`faster-whisper`) |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes secrets, tokens, and caches |

---

## Troubleshooting

**`invalid_grant` / authentication errors**
Delete `token.json` (or use **Settings → Clear Authentication Token**) and sign in again. Ensure `credentials.json` matches the OAuth client that originally issued the token.

**Missing API key**
Set `GOOGLE_API_KEY` before starting Streamlit, or enter it when the app prompts you on startup.

**Microphone not detected**
Ensure your system microphone permissions are granted for the terminal / Python process. On macOS: System Settings → Privacy & Security → Microphone.

**Whisper model download is slow**
The `medium` model (~460 MB) is downloaded once to `~/.cache/huggingface/`. After that, voice input works fully offline.

**Scope changes**
If you modify the OAuth scopes in code, delete `token.json` and re-authenticate.

---

## License

MIT License. See [LICENSE.md](LICENSE.md) for details.
