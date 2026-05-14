# Agentic workflow to manage Google calendar meetings, events, and tasks

import os
import json
import logging

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from google.genai.types import Tool
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal
from datetime import datetime, timezone, date as date_type

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/tasks",
]

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _log_api_call(method: str, **payload) -> None:
    """Log the complete JSON payload sent to a Google API call.

    `method` is a human-readable identifier like "calendar.events.insert".
    All keyword arguments are serialized as the request payload — pass the
    same kwargs you pass to the API method (e.g. calendarId, body, eventId).
    """
    logger.info(
        "Google API call: %s\n%s",
        method,
        json.dumps(payload, indent=2, default=str, sort_keys=True),
    )


client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
model_name = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash")

# --------------------------------------------------------------
# Step 1: Define the data models for each stage
# --------------------------------------------------------------

class KeyValuePair(BaseModel):
    key: str = Field(description="Key of the key-value pair")
    value: str = Field(description="Value of the key-value pair")

class EmailAddress(BaseModel):
    email: str = Field(description="Email address of the attendee")

class EventDateTime(BaseModel):
    """
    Represents the date and time of a Google Calendar event.
    """
    dateTime: Optional[str] = Field(
        None,
        description=(
            "The local date-time in 'YYYY-MM-DDTHH:MM:SS' format (no Z, no UTC offset). "
            "The timeZone field tells the Google Calendar API how to interpret this local time. "
            "NEVER append 'Z' or any UTC offset — doing so overrides timeZone and shifts the event to the wrong time."
        ),
    )
    timeZone: Optional[str] = Field(
        None,
        description=(
            "The time zone in which the time is specified (formatted as an IANA Time Zone Database name, e.g. 'America/Los_Angeles'). "
            "For recurring events this field is required and specifies the time zone in which the recurrence is expanded. "
            "For single events this field is optional and indicates a custom time zone for the event start/end."
        ),
    )

class CalendarEvent(BaseModel):
    """Determine whether the request describes a calendar meeting, event, or task."""
    description: str = Field(description="Text describing the request")
    is_calendar_event: bool = Field(
        description="True if this is a calendar meeting, event, or task request"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")

class CalendarRequestType(BaseModel):
    """Classify the action and item type of a calendar/task request."""
    description: str = Field(description="Text describing the request, stripped of action keywords")
    action: Literal["new", "modify", "delete", "other"] = Field(
        description="Action to take: new (create), modify (update), delete (remove), or other"
    )
    item_type: Literal["meeting", "event", "task", "birthday", "anniversary", "unknown"] = Field(
        description=(
            "Type of item: "
            "'meeting' = calendar event with at least one other attendee invited, "
            "'event' = personal calendar entry owned only by the calendar owner (no external attendees), "
            "'task' = a to-do item managed via Google Tasks (no specific calendar time slot required), "
            "'birthday' = a yearly recurring all-day birthday event for a specific person, "
            "'anniversary' = a yearly recurring all-day anniversary event (e.g. wedding, work, or other milestone)"
        )
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")

class ReminderOverride(BaseModel):
    """A single calendar notification rule."""
    method: Literal["email", "popup"] = Field(
        description="Notification method: 'email' sends an email, 'popup' shows an on-screen alert"
    )
    minutes: int = Field(
        description=(
            "Minutes before the event to deliver the notification. "
            "Examples: 10 = 10 minutes, 60 = 1 hour, 1440 = 1 day, 10080 = 1 week."
        )
    )

class EventReminders(BaseModel):
    """Reminder/notification settings for a calendar event."""
    useDefault: bool = Field(
        default=False,
        description="If True, use the calendar's default reminders. Set to False when specifying overrides."
    )
    overrides: list[ReminderOverride] = Field(
        default=[],
        description="Explicit reminder rules. Only used when useDefault is False."
    )

class NewEventDetails(BaseModel):
    """Details for creating a new calendar meeting or event"""
    summary: str = Field(description="Summary of the event")
    location: str = Field(description="Location of the event")
    description: str = Field(description="Description of the event")
    start: EventDateTime = Field(description="Start time object with fields dateTime and timeZone")
    end: EventDateTime = Field(description="End time object with fields dateTime and timeZone")
    recurrence: list[str] = Field(default=[], description="Recurrence rules")
    attendees: list[EmailAddress] = Field(
        description="List of attendee objects with fields email. Empty list for personal events."
    )
    reminders: Optional[EventReminders] = Field(
        default=None,
        description=(
            "Notification settings. Populate only when the user requests specific reminders. "
            "Set useDefault=False and list each override with method ('email' or 'popup') and minutes."
        )
    )

# TODO: Determine if this data model is needed
class ModifyEventDetails(BaseModel):
    """Details for modifying an existing calendar event"""
    summary: Optional[str] = Field(default=None, description="Summary of the event")
    location: Optional[str] = Field(description="Location of the event")
    description: Optional[str] = Field(description="Description of the event")
    start: Optional[EventDateTime] = Field(description="Start time object with fields dateTime and timeZone")
    end: Optional[EventDateTime] = Field(default=None, description="End time object with fields dateTime and timeZone")
    recurrence: Optional[list[str]] = Field(default=[], description="Recurrence rules")
    attendees: Optional[list[EmailAddress]] = Field(
        description="List of attendee objects with fields email"
    )

class EventsListParameters(BaseModel):
    """Parameters for listing calendar events"""
    calendarId: str = Field(description="Calendar ID")
    timeMin: Optional[str] = Field(default=None, description="Start time in RFC3339 format (e.g. '2026-03-29T00:00:00Z')")
    timeMax: Optional[str] = Field(default=None, description="End time in RFC3339 format (e.g. '2026-03-29T23:59:59Z')")
    singleEvents: bool = Field(default=False, description="Whether to return single events")
    orderBy: Optional[str] = Field(default=None, description="Order by")
    q: Optional[str] = Field(default=None, description="Query")

class AnnualEventDetails(BaseModel):
    """Extracted details from a birthday or anniversary event request."""
    summary: str = Field(description=(
        "Full event title. "
        "For a birthday use the format \"<Name>'s Birthday\" (e.g. \"Alice's Birthday\"). "
        "For an anniversary use the format \"<Name(s)>'s <Type> Anniversary\" "
        "(e.g. \"John and Jane's Wedding Anniversary\", \"Bob's Work Anniversary\")."
    ))
    date: str = Field(description="Event date in YYYY-MM-DD format (use current year if no year is specified)")
    event_type: Literal["birthday", "anniversary"] = Field(
        description="'birthday' for a person's birthday, 'anniversary' for any other yearly milestone"
    )

class TaskItem(BaseModel):
    """Details for creating or modifying a Google Task"""
    title: str = Field(description="Title of the task")
    notes: Optional[str] = Field(default=None, description="Notes or description of the task")
    due: Optional[str] = Field(
        default=None,
        description="Due date in RFC 3339 timestamp format, e.g. '2026-03-28T00:00:00.000Z'"
    )
    status: Optional[Literal["needsAction", "completed"]] = Field(
        default="needsAction", description="Status of the task"
    )

class CalendarResponse(BaseModel):
    """Final response format"""
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")
    calendar_link: Optional[str] = Field(description="Calendar link if applicable")

def _is_in_past(dt_str: str) -> bool:
    """Return True if the given date/datetime string represents a time that has already passed.

    Accepts:
      - "YYYY-MM-DDTHH:MM:SS" (local datetime, no offset — compared against local now)
      - "YYYY-MM-DD"           (all-day date — considered past if strictly before today)
    """
    dt_str = dt_str.strip()
    now = datetime.now()
    try:
        if "T" in dt_str:
            # Strip any accidental trailing Z or offset before comparing
            clean = dt_str[:19]
            event_dt = datetime.strptime(clean, "%Y-%m-%dT%H:%M:%S")
            return event_dt < now
        else:
            event_date = datetime.strptime(dt_str[:10], "%Y-%m-%d").date()
            return event_date < now.date()
    except ValueError:
        return False


# Invoke the GenAI (Gemini) model and return its response
def run_model(model_name, contents, config):
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config
    )
    return response

def parse_json_response(response) -> dict:
    """Parse JSON from a model response, handling markdown code fences and trailing text."""
    text = response.candidates[0].content.parts[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # drop the opening ```json line
        text = text.rsplit("```", 1)[0]  # drop the closing ```
    text = text.strip()
    obj, _ = json.JSONDecoder().raw_decode(text)
    return obj

# --------------------------------------------------------------
# Step 2: Define the functions to process calendar/task requests
# --------------------------------------------------------------

# Check if the user's description is a calendar meeting, event, or task
def check_if_calendar_event(description: str) -> CalendarEvent:
    """Check if the description is a calendar meeting, event, or task request."""
    logger.info("Checking if the description is a calendar/task request")
    logger.debug(f"Input text: {description}")

    config = types.GenerateContentConfig(
        system_instruction="""You are a calendar and task manager.
        Determine if the incoming request is for a calendar meeting, calendar event, or task.
        - A meeting is a calendar event that involves inviting at least one other person.
        - An event is a personal calendar entry (appointment, reminder, block time) with no external attendees.
        - A task is a to-do item that may or may not have a due date but does not occupy a calendar time slot.
        Return True for is_calendar_event if the request is for any of these three types.
        Return a confidence score between 0 and 1.
        """,
        response_mime_type="application/json",
        response_schema=CalendarEvent
    )

    contents = [
        types.Content(role="user", parts=[types.Part(text=description)])
    ]

    response = run_model(model_name, contents, config)
    response_json = parse_json_response(response)

    logger.info(
        f"Extraction complete - Is calendar/task request: {response_json['is_calendar_event']}, "
        f"Confidence: {response_json['confidence_score']:.2f}"
    )

    return response_json


# Classify the action and item type of the request
def determine_calendar_request_type(description: str) -> CalendarRequestType:
    """Classify the action (new/modify/delete) and item type (meeting/event/task)."""
    logger.info("Determining the type of calendar/task request")
    logger.debug(f"Input text: {description}")

    config = types.GenerateContentConfig(
        system_instruction="""You are a calendar and task manager.
        Given the user's request, determine:
        1. The ACTION: new (create something), modify (update something), delete (remove something), or other.
        2. The ITEM TYPE:
           - 'meeting': a calendar event where at least one other person is invited (emails or names of attendees are mentioned)
           - 'event': a personal calendar entry owned only by the calendar owner with no external attendees
             (e.g., a doctor's appointment, gym session, focus block, reminder with a time)
           - 'task': a to-do item managed via Google Tasks — no specific calendar time slot is required
             (e.g., "remind me to buy groceries", "add a task to submit the report")
           - 'birthday': a yearly recurring all-day birthday event for a specific person
             (e.g., "Create Alice's birthday on June 15", "Add John's birthday on March 3rd")
           - 'anniversary': a yearly recurring all-day anniversary event for a person or couple
             (e.g., "Add our wedding anniversary on July 4", "Create John and Jane's work anniversary on May 1")
        Also extract the cleaned description of the item, removing action keywords like "create", "schedule",
        "add", "delete", "modify", "update", "change".
        Return the action, item_type, cleaned description, and a confidence score between 0 and 1.
        """,
        response_mime_type="application/json",
        response_schema=CalendarRequestType
    )

    contents = [
        types.Content(role="user", parts=[types.Part(text=description)])
    ]

    response = run_model(model_name, contents, config)
    response_json = parse_json_response(response)

    logger.info(
        f"Extraction complete - Action: {response_json['action']}, "
        f"Item type: {response_json['item_type']}, "
        f"Confidence: {response_json['confidence_score']:.2f}"
    )

    return response_json


# Get a list of calendar events given the user's description
def get_calendar_events(credentials, calendar_id, description: str) -> list:
    """Get a list of calendar events (meetings or personal events)."""
    logger.info("Getting a list of calendar events")
    logger.debug(f"Input text: {description}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    config = types.GenerateContentConfig(
        system_instruction=f"""You are an expert Google calendar manager.
        Given the {date_context} build a JSON object to fetch the Google calendar events referenced in the description.
        If no start date is specified then use today at 12:00 AM as timeMin.
        Do not create a default timeMax. Only populate timeMax if the description specifies an end date.
        The q field should contain the text from the description that would be in the summary of the Google calendar event.
        Return ONLY the relevant fields from the following list in JSON format:
        - calendarId: string
        - timeMin: datetime
        - timeMax: datetime
        - singleEvents: bool
        - orderBy: string
        - q: string
        Do not include any other fields or properties.
        """,
        response_mime_type="application/json",
        response_schema=EventsListParameters
    )

    contents = [
        types.Content(role="user", parts=[types.Part(text=description)])
    ]

    response = run_model(model_name, contents, config)
    response_json = parse_json_response(response)
    logger.info(f"Events List Parameters: {response_json}")

    list_kwargs = {
        "calendarId": response_json["calendarId"],
        "timeMin": response_json.get("timeMin"),
        "timeMax": response_json.get("timeMax"),
        "singleEvents": response_json.get("singleEvents", False),
        "orderBy": response_json.get("orderBy"),
        "q": response_json.get("q"),
    }
    _log_api_call("calendar.events.list", **list_kwargs)

    try:
        service = build("calendar", "v3", credentials=credentials)
        events_result = service.events().list(**list_kwargs).execute()
        events = events_result.get("items", [])
        logger.info(f"Found {len(events)} event(s)")
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        return []

    return events


# Get a list of Google Tasks matching the description
def get_tasks(credentials, description: str) -> list:
    """Get a list of Google Tasks from the default task list."""
    logger.info("Getting a list of Google Tasks")
    logger.debug(f"Input text: {description}")

    list_kwargs = {
        "tasklist": "@default",
        "showCompleted": False,
        "showDeleted": False,
    }
    _log_api_call("tasks.tasks.list", **list_kwargs)

    try:
        service = build("tasks", "v1", credentials=credentials)
        tasks_result = service.tasks().list(**list_kwargs).execute()
        tasks = tasks_result.get("items", [])
        logger.info(f"Found {len(tasks)} task(s) total")

        # Filter by description if it's not a generic "all tasks" request
        if description and description.lower() not in ("all tasks", "all", ""):
            filtered = [
                t for t in tasks
                if description.lower() in t.get("title", "").lower()
                or description.lower() in t.get("notes", "").lower()
            ]
            logger.info(f"Filtered to {len(filtered)} task(s) matching '{description}'")
            return filtered

        return tasks
    except HttpError as error:
        logger.error(f"An error occurred fetching tasks: {error}")
        return []


# Create a new calendar meeting or personal event
def create_new_event(credentials, calendar_id, description: str, item_type: str = "event", reminders_override: Optional[dict] = None) -> CalendarResponse:
    """Create a new calendar meeting (with attendees) or personal event (no attendees)."""
    logger.info(f"Creating a new calendar {item_type}")
    logger.debug(f"Input text: {description}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    if item_type == "meeting":
        attendee_instruction = (
            "This is a MEETING — populate the attendees list with all people mentioned in the description. "
            "At least one attendee email must be included."
        )
    else:
        attendee_instruction = (
            "This is a personal EVENT — the attendees list must be empty ([])."
        )

    config = types.GenerateContentConfig(
        system_instruction=f"""You are a calendar event manager.
        Given the {date_context} create a new calendar entry based on the description.
        {attendee_instruction}

        IMPORTANT — dateTime format rules:
        - Express dateTime as local time in the format "YYYY-MM-DDTHH:MM:SS" (no Z, no UTC offset).
        - Always set the timeZone field to the correct IANA timezone name (e.g. "America/Los_Angeles").
        - NEVER append "Z" or any UTC offset (e.g. "+00:00", "-07:00") to dateTime. The timeZone field
          tells the Google Calendar API how to interpret the local time. Adding "Z" overrides timeZone
          and will place the event at the wrong time.

        Return ONLY these exact fields in JSON format:
        - summary: string
        - location: string
        - description: string
        - start: object with dateTime (local, no Z) and timeZone (IANA name)
        - end: object with dateTime (local, no Z) and timeZone (IANA name)
        - recurrence: array of strings
        - attendees: array of objects with email field
        - reminders: object with useDefault (bool) and overrides (array of objects with method and minutes).
          Only populate reminders if the user explicitly requests notifications; otherwise omit it.
          Example: {{"useDefault": false, "overrides": [{{"method": "email", "minutes": 1440}}, {{"method": "popup", "minutes": 30}}]}}
        Do not include any other fields or properties.
        """,
        response_mime_type="application/json",
        response_schema=NewEventDetails
    )

    contents = [
        types.Content(role="user", parts=[types.Part(text=description)])
    ]

    response = run_model(model_name, contents, config)
    response_json = parse_json_response(response)

    # Reject events scheduled in the past
    start_dt = response_json.get("start", {}).get("dateTime") or response_json.get("start", {}).get("date")
    if start_dt and _is_in_past(start_dt):
        logger.warning(f"Refused to create past {item_type}: start={start_dt}")
        return CalendarResponse(
            success=False,
            message=f"Cannot create a {item_type} in the past (start: {start_dt}). Please provide a future date and time.",
            calendar_link=None
        )

    # UI-supplied reminders take precedence over LLM-extracted reminders
    if reminders_override is not None:
        response_json["reminders"] = reminders_override
        logger.info(f"Reminders override applied: {reminders_override}")

    _log_api_call("calendar.events.insert", calendarId=calendar_id, body=response_json)

    try:
        service = build("calendar", "v3", credentials=credentials)
        event = service.events().insert(calendarId=calendar_id, body=response_json).execute()
        logger.info(f"New calendar {item_type} created: {event.get('htmlLink')}")
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        return CalendarResponse(
            success=False,
            message=f"An error occurred: {error}",
            calendar_link=None
        )

    attendees_info = response_json.get("attendees", [])
    attendee_str = f" with {attendees_info}" if attendees_info else ""
    return CalendarResponse(
        success=True,
        message=f"New {item_type} '{response_json['summary']}' created for {response_json['start']['dateTime']}{attendee_str}",
        calendar_link=event.get("htmlLink")
    )


# Create a new Google Task
def create_task(credentials, description: str) -> CalendarResponse:
    """Create a new Google Task in the default task list."""
    logger.info("Creating a new Google Task")
    logger.debug(f"Input text: {description}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    config = types.GenerateContentConfig(
        system_instruction=f"""You are a task manager.
        Given the {date_context} create a new task based on the description.
        Return ONLY these fields in JSON format:
        - title: string (required — the task name)
        - notes: string (optional — additional details or context)
        - due: string (optional — due date in RFC 3339 format, e.g. "2026-03-28T00:00:00.000Z")
        - status: string ("needsAction" or "completed", default "needsAction")
        Do not include any other fields.
        """,
        response_mime_type="application/json",
        response_schema=TaskItem
    )

    contents = [
        types.Content(role="user", parts=[types.Part(text=description)])
    ]

    response = run_model(model_name, contents, config)
    response_json = parse_json_response(response)

    _log_api_call("tasks.tasks.insert", tasklist="@default", body=response_json)

    try:
        service = build("tasks", "v1", credentials=credentials)
        task = service.tasks().insert(tasklist="@default", body=response_json).execute()
        logger.info(f"Task created: {task.get('title')}")
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        return CalendarResponse(
            success=False,
            message=f"An error occurred: {error}",
            calendar_link=None
        )

    due_str = f" due {response_json['due']}" if response_json.get("due") else ""
    return CalendarResponse(
        success=True,
        message=f"Task '{response_json['title']}' created successfully{due_str}",
        calendar_link=None
    )


# Create a new birthday event that repeats yearly
def create_annual_event(credentials, calendar_id, description: str) -> CalendarResponse:
    """Create a yearly all-day birthday or anniversary event with default email (1 day) and popup (15 min) reminders."""
    logger.info("Creating an annual event (birthday or anniversary)")
    logger.debug(f"Input text: {description}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    config = types.GenerateContentConfig(
        system_instruction=f"""You are a calendar assistant. {date_context}
        Extract the event summary, date, and event type (birthday or anniversary) from the description.
        - For a birthday, format the summary as "<Name>'s Birthday".
        - For an anniversary, format the summary as "<Name(s)>'s <Type> Anniversary"
          (e.g. "John and Jane's Wedding Anniversary", "Bob's Work Anniversary").
        Return the date in YYYY-MM-DD format.
        When only a month and day are given (no year), use the current year UNLESS that date
        has already passed this year, in which case use next year.
        """,
        response_mime_type="application/json",
        response_schema=AnnualEventDetails
    )

    contents = [types.Content(role="user", parts=[types.Part(text=description)])]

    response = run_model(model_name, contents, config)
    details = parse_json_response(response)

    summary = details["summary"]
    date_str = details["date"]  # YYYY-MM-DD
    event_type = details["event_type"]  # "birthday" or "anniversary"

    # For annual events the user typically omits the year (e.g. "June 15").
    # If the LLM resolved that to a past date in the current year, advance it
    # to the same month/day next year so the first Google Calendar occurrence
    # is the upcoming one.
    if _is_in_past(date_str):
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d")
            date_str = parsed.replace(year=parsed.year + 1).strftime("%Y-%m-%d")
            logger.info(f"Annual event date was in the past; advanced to {date_str}")
        except ValueError:
            pass  # leave date_str as-is; the check below will catch it

    # After the year-advance above, a date is only still in the past if the
    # user explicitly supplied a past year (e.g. "anniversary on July 4 2020").
    if _is_in_past(date_str):
        logger.warning(f"Refused to create past annual event: date={date_str}")
        return CalendarResponse(
            success=False,
            message=f"Cannot create \"{summary}\" with a date in the past ({date_str}). Please provide a future date.",
            calendar_link=None
        )

    # All-day events require an end date one day after the start
    from datetime import timedelta
    start_date = datetime.strptime(date_str, "%Y-%m-%d")
    end_date_str = (start_date + timedelta(days=1)).strftime("%Y-%m-%d")

    event_body = {
        "summary": summary,
        "start": {"date": date_str},
        "end": {"date": end_date_str},
        "recurrence": ["RRULE:FREQ=YEARLY"],
        "transparency": "transparent",  # show as free for all annual events
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "email", "minutes": 15},  # 15 minutes before
                {"method": "popup", "minutes": 15},    # 15 minutes before
            ]
        }
    }

    # "birthday" is a recognized eventType in the Google Calendar API; anniversaries use the default.
    if event_type == "birthday":
        event_body["eventType"] = "birthday"
        event_body["visibility"] = "private"  # required by the API for birthday event type

    _log_api_call("calendar.events.insert", calendarId=calendar_id, body=event_body)

    try:
        service = build("calendar", "v3", credentials=credentials)
        event = service.events().insert(calendarId=calendar_id, body=event_body).execute()
        logger.info(f"Annual event created: {event.get('htmlLink')}")
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        return CalendarResponse(
            success=False,
            message=f"An error occurred: {error}",
            calendar_link=None
        )

    return CalendarResponse(
        success=True,
        message=f"\"{summary}\" created for {date_str}, repeating yearly",
        calendar_link=event.get("htmlLink")
    )


# Delete an existing calendar event by ID
def delete_event_by_id(credentials, calendar_id, event_id: str) -> CalendarResponse:
    """Delete an existing calendar event by ID."""
    logger.info("Deleting an existing calendar event by ID")
    logger.info(f"Event ID: {event_id}")

    _log_api_call("calendar.events.delete", calendarId=calendar_id, eventId=event_id)

    try:
        service = build("calendar", "v3", credentials=credentials)
        service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
        logger.info(f"Event {event_id} deleted")
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        return CalendarResponse(
            success=False,
            message=f"An error occurred: {error}",
            calendar_link=None
        )

    return CalendarResponse(
        success=True,
        message=f"Event {event_id} deleted",
        calendar_link=None
    )


# Delete one or more calendar events given the user's description
def delete_event(credentials, calendar_id, description: str, all: bool = False) -> CalendarResponse:
    """Delete one or more calendar meetings or events matching the description."""
    logger.info("Deleting calendar event(s)")
    logger.info(f"Input text: {description}")

    events = get_calendar_events(credentials, calendar_id, description)
    logger.info(f"Found {len(events)} event(s)")

    if not events:
        return CalendarResponse(
            success=False,
            message=f"No events found matching '{description}'",
            calendar_link=None
        )

    calResponseMessage = ""

    if all:
        for event in events:
            calResponse = delete_event_by_id(credentials, calendar_id, event["id"])
            if calResponse.success:
                calResponseMessage += f"Deleted: {event['summary']} ({event['start'].get('dateTime', event['start'].get('date'))})\n"
            else:
                calResponseMessage += f"Error deleting {event['id']}: {calResponse.message}\n"
    else:
        event = events[0]
        calResponse = delete_event_by_id(credentials, calendar_id, event["id"])
        if calResponse.success:
            calResponseMessage = f"Deleted: {event['summary']} ({event['start'].get('dateTime', event['start'].get('date'))})"
        else:
            calResponseMessage = f"Error deleting {event['id']}: {calResponse.message}"

    return CalendarResponse(
        success=True,
        message=calResponseMessage,
        calendar_link=None
    )


# Delete one or more Google Tasks matching the description
def delete_task(credentials, description: str, all: bool = False) -> CalendarResponse:
    """Delete one or more Google Tasks matching the description."""
    logger.info("Deleting Google Task(s)")
    logger.info(f"Input text: {description}")

    tasks = get_tasks(credentials, description)
    if not tasks:
        return CalendarResponse(
            success=False,
            message=f"No tasks found matching '{description}'",
            calendar_link=None
        )

    try:
        service = build("tasks", "v1", credentials=credentials)
        tasks_to_delete = tasks if all else [tasks[0]]
        deleted_titles = []
        for task in tasks_to_delete:
            _log_api_call("tasks.tasks.delete", tasklist="@default", task=task["id"])
            service.tasks().delete(tasklist="@default", task=task["id"]).execute()
            deleted_titles.append(task.get("title", task["id"]))
            logger.info(f"Task {task['id']} deleted")
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        return CalendarResponse(
            success=False,
            message=f"An error occurred: {error}",
            calendar_link=None
        )

    return CalendarResponse(
        success=True,
        message=f"Deleted task(s): {', '.join(deleted_titles)}",
        calendar_link=None
    )


# Modify an existing calendar event given the user's description
def modify_event(credentials, calendar_id, description: str, reminders_override: Optional[dict] = None) -> CalendarResponse:
    """Modify an existing calendar meeting or event."""
    logger.info("Modifying an existing calendar event")
    logger.debug(f"Input text: {description}")

    events = get_calendar_events(credentials, calendar_id, description)
    logger.info(f"Found {len(events)} event(s)")

    if len(events) == 0:
        return CalendarResponse(
            success=False,
            message=f"No events found for the description '{description}'",
            calendar_link=None
        )
    elif len(events) > 1:
        return CalendarResponse(
            success=False,
            message=f"Multiple events found for '{description}'. Please be more specific.",
            calendar_link=None
        )

    event = events[0]
    logger.info(f"Event to modify: {event['id']}: {event['summary']}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    config = types.GenerateContentConfig(
        system_instruction=f"""You are a Google Calendar manager well versed in the Google Calendar API.
        The user is requesting a modification to an existing calendar event '{event}' given that {date_context}.
        Starting with the current calendar event, create a JSON object to modify the event based on the user's description.
        Update ONLY the fields that are to be modified.

        IMPORTANT — dateTime format rules:
        - Express dateTime as local time in the format "YYYY-MM-DDTHH:MM:SS" (no Z, no UTC offset).
        - Always set the timeZone field to the correct IANA timezone name (e.g. "America/Los_Angeles").
        - NEVER append "Z" or any UTC offset to dateTime. Adding "Z" overrides timeZone and will place
          the event at the wrong local time.

        Return ONLY the fields to be modified in JSON format:
        - summary: string
        - location: string
        - description: string
        - start: object with dateTime (local, no Z) and timeZone (IANA name)
        - end: object with dateTime (local, no Z) and timeZone (IANA name)
        - recurrence: array of strings
        - attendees: array of objects with email field
        - reminders: object with useDefault (bool) and overrides (array of objects with method and minutes).
          Only include reminders if the user explicitly requests a notification change.
          Example: {{"useDefault": false, "overrides": [{{"method": "email", "minutes": 1440}}, {{"method": "popup", "minutes": 30}}]}}
        Do not include any other fields or properties.
        """,
        response_mime_type="application/json",
        response_schema=NewEventDetails
    )

    contents = [
        types.Content(role="user", parts=[types.Part(text=description)])
    ]

    response = run_model(model_name, contents, config)
    response_json = parse_json_response(response)

    # If the original event is an all-day event, ensure start/end use "date" not "dateTime"
    is_all_day = "date" in event.get("start", {}) and "dateTime" not in event.get("start", {})
    if is_all_day:
        for field in ("start", "end"):
            if field in response_json and "dateTime" in response_json[field]:
                dt_str = response_json[field]["dateTime"]
                date_only = dt_str[:10]  # extract YYYY-MM-DD
                response_json[field] = {"date": date_only}

    # Reject modifications that would move the event to a past date/time
    new_start = response_json.get("start", {})
    new_start_dt = new_start.get("dateTime") or new_start.get("date")
    if new_start_dt and _is_in_past(new_start_dt):
        logger.warning(f"Refused to modify event to past time: start={new_start_dt}")
        return CalendarResponse(
            success=False,
            message=f"Cannot reschedule \"{event['summary']}\" to a time in the past ({new_start_dt}). Please provide a future date and time.",
            calendar_link=None
        )

    # UI-supplied reminders take precedence over LLM-extracted reminders
    if reminders_override is not None:
        response_json["reminders"] = reminders_override
        logger.info(f"Reminders override applied: {reminders_override}")

    _log_api_call("calendar.events.patch", calendarId=calendar_id, eventId=event["id"], body=response_json)

    try:
        service = build("calendar", "v3", credentials=credentials)
        updated_event = service.events().patch(
            calendarId=calendar_id, eventId=event["id"], body=response_json
        ).execute()
        logger.info(f"Event {event['id']} successfully modified")
    except HttpError as error:
        logger.error(f"An error occurred while modifying the event ({event['id']}): {error}")
        return CalendarResponse(
            success=False,
            message=f"An error occurred while modifying the event ({event['id']}): {error}",
            calendar_link=None
        )

    return CalendarResponse(
        success=True,
        message=f"Modified: {events[0]['summary']} ({events[0]['start'].get('dateTime', events[0]['start'].get('date'))})",
        calendar_link=None
    )


# Modify an existing Google Task given the user's description
def modify_task(credentials, description: str) -> CalendarResponse:
    """Modify an existing Google Task."""
    logger.info("Modifying an existing Google Task")
    logger.debug(f"Input text: {description}")

    tasks = get_tasks(credentials, description)
    if not tasks:
        return CalendarResponse(
            success=False,
            message=f"No tasks found matching '{description}'",
            calendar_link=None
        )
    if len(tasks) > 1:
        return CalendarResponse(
            success=False,
            message=f"Multiple tasks found matching '{description}'. Please be more specific.",
            calendar_link=None
        )

    task = tasks[0]
    logger.info(f"Task to modify: {task['id']}: {task.get('title')}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    config = types.GenerateContentConfig(
        system_instruction=f"""You are a task manager.
        The user wants to modify the existing task '{task}' given that {date_context}.
        Based on the description, create a JSON object with only the fields that need to change:
        - title: string
        - notes: string
        - due: string (RFC 3339 format, e.g. "2026-03-28T00:00:00.000Z")
        - status: string ("needsAction" or "completed")
        Do not include fields that are not being changed.
        """,
        response_mime_type="application/json",
        response_schema=TaskItem
    )

    contents = [
        types.Content(role="user", parts=[types.Part(text=description)])
    ]

    response = run_model(model_name, contents, config)
    response_json = parse_json_response(response)

    _log_api_call("tasks.tasks.patch", tasklist="@default", task=task["id"], body=response_json)

    try:
        service = build("tasks", "v1", credentials=credentials)
        service.tasks().patch(
            tasklist="@default", task=task["id"], body=response_json
        ).execute()
        logger.info(f"Task {task['id']} modified")
    except HttpError as error:
        logger.error(f"An error occurred: {error}")
        return CalendarResponse(
            success=False,
            message=f"An error occurred: {error}",
            calendar_link=None
        )

    return CalendarResponse(
        success=True,
        message=f"Task '{task.get('title')}' modified successfully",
        calendar_link=None
    )


# ---------------------------------------------------------------------------------
# Step 3: Route the calendar/task request to the appropriate handler
# ---------------------------------------------------------------------------------

def process_calendar_request(credentials, calendar_id, user_input: str, reminders_override: Optional[dict] = None) -> Optional[CalendarResponse]:
    """Process an incoming calendar or task request and route to the correct handler."""
    logger.info(f"Processing request: {user_input}")

    # Step 1: Check if this is a calendar/task request at all
    is_calendar_event = check_if_calendar_event(user_input)
    if not (is_calendar_event["is_calendar_event"] and is_calendar_event["confidence_score"] > 0.7):
        logger.warning("Request is not a recognized calendar or task request")
        return None

    # Step 2: Classify action and item type
    request_type = determine_calendar_request_type(user_input)
    logger.info(f"Request type: {request_type}")

    action = request_type.get("action", "other")
    item_type = request_type.get("item_type", "unknown")
    description = request_type.get("description", user_input)
    confidence = request_type.get("confidence_score", 0)

    if confidence <= 0.7:
        logger.warning(f"Low confidence ({confidence:.2f}), skipping")
        return None

    # Step 3: Route to the appropriate handler
    if action == "new":
        if item_type == "task":
            return create_task(credentials, description)
        elif item_type in ("birthday", "anniversary"):
            return create_annual_event(credentials, calendar_id, description)
        else:
            # Both "meeting" and "event" use the calendar API; item_type controls attendee handling
            return create_new_event(credentials, calendar_id, description, item_type=item_type, reminders_override=reminders_override)

    elif action == "modify":
        if item_type == "task":
            return modify_task(credentials, description)
        else:
            return modify_event(credentials, calendar_id, description, reminders_override=reminders_override)

    elif action == "delete":
        if item_type == "task":
            return delete_task(credentials, description)
        else:
            return delete_event(credentials, calendar_id, description)

    else:
        logger.warning(f"Unsupported action '{action}' for item type '{item_type}'")
        return None


# --------------------------------------------------------------
# Step 4: Define the main function to run the calendar agent
# --------------------------------------------------------------

def main():
    """Run the calendar agent interactively from the command line."""

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    print("\n=== Google Calendar Agent ===")
    print("Describe a meeting, event, or task you want to create, modify, or delete.")
    print("Examples:")
    print("  'Schedule a team meeting with John (john@email.com) tomorrow at 2 PM for 1 hour'")
    print("  'Add a dentist appointment on Friday at 10 AM'")
    print("  'Add a task to review the Q1 report by end of week'")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("Enter request: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter a valid request.")
            continue

        print(f"\nProcessing: {user_input}")
        result = process_calendar_request(creds, "primary", user_input)

        if result and result.success:
            print(f"✅ {result.message}")
        else:
            msg = result.message if result else "Could not process request."
            print(f"❌ {msg} Please try again with a clearer description.")

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
