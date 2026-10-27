"""Lookups over the structured course data (courses.csv, timetable.csv, faculty.csv, student_profile.json)."""

from __future__ import annotations

from helpdesk import catalog, config


def _schedule_text(slots: list[dict]) -> str:
    return ", ".join(f"{m['day']} {m['start']}-{m['end']}" for m in slots)


def _course_summary(course: dict) -> dict:
    teacher = catalog.instructor(course)
    summary = {
        "code": course["code"],
        "title": course["title"],
        "department": course["department"],
        "level": int(course["level"]),
        "category": course["category"],
        "credits": float(course["credits"]),
        "ltp": course["ltp"],
        "semester": course["semester"],
        "slot": course["slot"] or "not assigned a slot letter (scheduled at fixed times)",
        "lectures": _schedule_text(catalog.meetings(course)) or "by arrangement",
        "room": course["room"],
        "instructor": teacher["name"],
        "offered_to": course["offered_to"],
    }
    if course["lab_slot"]:
        summary["lab"] = f"slot {course['lab_slot']}: {_schedule_text(catalog.lab_meetings(course))}"
    return summary


def find_course(course: str) -> dict:
    """Looks up a course offered this semester by its code (e.g. "CSL7570") or title
    (e.g. "Autonomous Systems"): slot, lecture days and times, room, credits, level, instructor.

    Args:
        course: Course code or course title.

    Returns:
        dict: `status` "success" with `courses` (one or more matches), or "not_found".
    """
    matches = catalog.find_courses(course)
    if not matches:
        return {"status": "not_found",
                "message": f"No course matching '{course}' is offered in {config.CURRENT_SEM}."}
    return {"status": "success", "courses": [_course_summary(c) for c in matches],
            "source": f"{catalog.COURSES_SOURCE}; {catalog.SLOTS_SOURCE}"}


def list_courses(department: str | None = None, level: int | None = None,
                 category: str | None = None) -> dict:
    """Lists and counts the courses floated this semester, optionally filtered. Use it for questions
    like "How many 400 level electives has the Humanities department floated this semester?".

    Args:
        department: Department or school, e.g. "Humanities", "Computer Science", "Economics".
        level: Course level, e.g. 400 for 400-level courses (also accepts 4).
        category: "elective", "core" or "project". "elective" includes humanities electives.

    Returns:
        dict: `status`, `count`, and `courses` (code, title, department, level, category).
    """
    pool = [c for c in catalog.courses() if c["semester"] == config.CURRENT_SEM]
    applied = {"semester": config.CURRENT_SEM}

    if department:
        names = catalog.match_department(department)
        if not names:
            departments = sorted({c["department"] for c in pool})
            return {"status": "unknown_department",
                    "message": f"No department matches '{department}'.", "departments": departments}
        pool = [c for c in pool if c["department"] in names]
        applied["department"] = names
    if level:
        # Some models send the level as a string ("400", "400-level"); ADK passes it through unconverted.
        digits = "".join(ch for ch in str(level) if ch.isdigit())
        if not digits:
            return {"status": "error", "message": f"Level must be a number such as 400, not '{level}'."}
        level = int(digits)
        level = level * 100 if level < 10 else level
        pool = [c for c in pool if int(c["level"]) == level]
        applied["level"] = level
    if category:
        key = category.lower().strip().rstrip("s")
        pool = [c for c in pool if key in c["category"]]
        applied["category"] = key

    return {
        "status": "success",
        "filters": applied,
        "count": len(pool),
        "courses": [{k: c[k] for k in ("code", "title", "department", "level", "category")} for c in pool],
        "source": catalog.COURSES_SOURCE,
    }


def get_slot_timetable(slot: str) -> dict:
    """Gives the days and times of a timetable slot letter (e.g. "A", "F") or lab slot (e.g. "L3", "M6").

    Args:
        slot: Slot name.

    Returns:
        dict: `status` and the slot's `meetings` (day, start, end).
    """
    slot = slot.strip().upper().removeprefix("SLOT").strip()
    rows = [t for t in catalog.timetable() if t["slot"] == slot]
    if not rows:
        known = sorted({t["slot"] for t in catalog.timetable()})
        return {"status": "not_found", "message": f"There is no slot '{slot}'.", "slots": known}
    return {"status": "success", "slot": slot, "type": rows[0]["type"],
            "meetings": [{"day": t["day"], "start": t["start"], "end": t["end"]} for t in rows],
            "source": catalog.SLOTS_SOURCE}


def get_my_schedule(day: str | None = None) -> dict:
    """Gives the student's own registered courses this semester with their lecture times and rooms.
    Use it for "my classes", "my timetable" or "what do I have on Tuesday".

    Args:
        day: Optional weekday to filter by, e.g. "Tuesday" or "Tue".

    Returns:
        dict: `status` and `classes` sorted by day and time.
    """
    profile = catalog.profile()
    by_code = {c["code"]: c for c in catalog.courses()}
    wanted = catalog.DAYS.get(day.strip().lower()[:3]) if day else None
    if day and not wanted:
        return {"status": "error", "message": f"'{day}' is not a weekday."}

    order = list(catalog.DAYS.values())
    classes = []
    for code in profile["registered_courses"]:
        course = by_code[code]
        for m in catalog.meetings(course):
            if wanted in (None, m["day"]):
                classes.append({**m, "code": code, "title": course["title"], "room": course["room"]})
    classes.sort(key=lambda m: (order.index(m["day"]), m["start"]))
    return {"status": "success", "semester": profile["academic_semester"], "day": wanted or "all",
            "registered_courses": [f"{c} {by_code[c]['title']}" for c in profile["registered_courses"]],
            "classes": classes, "source": f"{catalog.PROFILE_SOURCE}; {catalog.COURSES_SOURCE}"}


def get_instructor_contact(course: str) -> dict:
    """Finds the instructor of a course offered this semester and their official email address.

    Args:
        course: Course code (e.g. "CSL7570") or course title.

    Returns:
        dict: `status` and, per matching course, the instructor's name, designation, department and email.
    """
    matches = catalog.find_courses(course)
    if not matches:
        return {"status": "not_found",
                "message": f"No course matching '{course}' is offered in {config.CURRENT_SEM}."}
    return {
        "status": "success",
        "instructors": [
            {"course": f"{c['code']} {c['title']}", **{k: catalog.instructor(c)[k]
                                                       for k in ("name", "designation", "department", "email")}}
            for c in matches
        ],
        "source": f"{catalog.COURSES_SOURCE}; {catalog.FACULTY_SOURCE}",
    }
