"""Read-only access to the structured data: courses, slot timetable, faculty and the student profile."""

from __future__ import annotations

import csv
import difflib
import json
import re
from functools import cache

from helpdesk import config

# Source labels used in citations for answers built from the structured data
COURSES_SOURCE = "Course Offerings, Semester I 2026-27 (courses.csv)"
SLOTS_SOURCE = "Slot System for B.Tech. 2nd year onwards, AY 2026-27 Semester I (timetable.csv)"
FACULTY_SOURCE = "IIT Jodhpur Faculty Directory (faculty.csv)"
PROFILE_SOURCE = "Student registration record (student_profile.json)"

DAYS = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday",
        "sat": "Saturday", "sun": "Sunday"}

# Common ways students refer to departments -> a substring of the official department name
DEPARTMENT_ALIASES = {
    "hss": "humanities", "sola": "humanities", "liberal arts": "humanities", "humanity": "humanities",
    "cse": "computer science", "cs": "computer science", "computer science": "computer science",
    "ee": "electrical", "ece": "electronics",
    "sme": "management", "management": "management",
    "saide": "artificial intelligence and data science", "aide": "artificial intelligence and data science",
    "ai": "artificial intelligence and data science",
    "idrp": "robotics", "rms": "robotics",
    "math": "mathematics", "maths": "mathematics",
    "environment": "environment", "ese": "environment",
}


def _read_csv(name: str) -> list[dict]:
    with (config.STRUCTURED_DIR / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@cache
def courses() -> list[dict]:
    return _read_csv("courses.csv")


@cache
def timetable() -> list[dict]:
    return _read_csv("timetable.csv")


@cache
def faculty() -> dict[str, dict]:
    return {f["instructor_id"]: f for f in _read_csv("faculty.csv")}


@cache
def profile() -> dict:
    return json.loads((config.STRUCTURED_DIR / "student_profile.json").read_text(encoding="utf-8"))


def normalize_code(text: str) -> str:
    return re.sub(r"[\s\-]", "", text).upper()


def meetings(course: dict) -> list[dict]:
    """Lecture meetings of a course: from its slot letter, or from its explicit `meetings` column."""
    if course["slot"]:
        return [{"day": t["day"], "start": t["start"], "end": t["end"]}
                for t in timetable() if t["slot"] == course["slot"]]
    out = []
    for part in filter(None, (p.strip() for p in course["meetings"].split(";"))):
        day, start, end = re.fullmatch(r"(\w{3}) (\d\d:\d\d)-(\d\d:\d\d)", part).groups()
        out.append({"day": DAYS[day.lower()], "start": start, "end": end})
    return out


def lab_meetings(course: dict) -> list[dict]:
    return [{"day": t["day"], "start": t["start"], "end": t["end"]}
            for t in timetable() if course["lab_slot"] and t["slot"] == course["lab_slot"]]


def find_courses(query: str, semester: str | None = config.CURRENT_SEM) -> list[dict]:
    """Match a course by code (exact) or title (exact, substring, then fuzzy). Best matches first."""
    pool = [c for c in courses() if semester is None or c["semester"] == semester]
    code = normalize_code(query)
    exact_code = [c for c in pool if c["code"] == code]
    if exact_code:
        return exact_code
    q = query.lower().strip()
    exact = [c for c in pool if c["title"].lower() == q]
    if exact:
        return exact
    contains = [c for c in pool if q in c["title"].lower() or c["title"].lower() in q]
    if contains:
        return contains
    titles = {c["title"].lower(): c for c in pool}
    return [titles[t] for t in difflib.get_close_matches(q, titles, n=3, cutoff=0.6)]


def match_department(query: str) -> list[str]:
    """Official department names matching a user's wording, e.g. 'Humanities' or 'HSS'."""
    q = re.sub(r"\b(department|dept|school|of|the)\b", " ", query.lower())
    q = re.sub(r"\s+", " ", q).strip()
    key = DEPARTMENT_ALIASES.get(q, q)
    names = sorted({c["department"] for c in courses()})
    return [n for n in names if key in n.lower()]


def instructor(course: dict) -> dict:
    return faculty()[course["instructor_id"]]
