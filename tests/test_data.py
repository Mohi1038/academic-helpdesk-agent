"""Integrity checks for the corpus manifest and the structured CSVs."""

import csv
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRUCTURED = ROOT / "data" / "structured"
CURRENT_SEM = "2026-27-I"
DAYS = {"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Fri": "Friday"}


def read(name: str) -> list[dict]:
    with (STRUCTURED / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


courses, faculty, timetable = read("courses.csv"), read("faculty.csv"), read("timetable.csv")
current = [c for c in courses if c["semester"] == CURRENT_SEM]


def lecture_meetings(course: dict) -> list[tuple[str, str, str]]:
    """(day, start, end) for every lecture of a course, from its slot or its explicit meetings."""
    if course["slot"]:
        return [(t["day"], t["start"], t["end"]) for t in timetable if t["slot"] == course["slot"]]
    out = []
    for part in filter(None, (p.strip() for p in course["meetings"].split(";"))):
        day, start, end = re.fullmatch(r"(\w{3}) (\d\d:\d\d)-(\d\d:\d\d)", part).groups()
        out.append((DAYS[day], start, end))
    return out


def overlap(a: tuple, b: tuple) -> bool:
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def clashes(group: list[dict]) -> list[tuple[str, str]]:
    found = []
    for x, y in itertools.combinations(group, 2):
        if any(overlap(m, n) for m in lecture_meetings(x) for n in lecture_meetings(y)):
            found.append((x["code"], y["code"]))
    return found


def test_manifest_files_exist_and_count_in_range():
    with (ROOT / "data" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert 20 <= len(rows) <= 30
    assert len({r["doc_id"] for r in rows}) == len(rows)
    assert len({r["sha256"] for r in rows}) == len(rows), "duplicate documents"
    for r in rows:
        assert (ROOT / r["file"]).exists(), r["file"]


def test_course_codes_unique():
    codes = [c["code"] for c in courses]
    assert len(codes) == len(set(codes))


def test_course_level_matches_code():
    for c in courses:
        assert c["code"][3] == c["level"][0], c["code"]


def test_references_resolve():
    ids = {f["instructor_id"] for f in faculty}
    slots = {t["slot"] for t in timetable}
    for c in courses:
        assert c["instructor_id"] in ids, c["code"]
        for s in (c["slot"], c["lab_slot"]):
            assert not s or s in slots, (c["code"], s)


def test_every_scheduled_course_has_meetings():
    for c in courses:
        if c["category"] != "project":
            assert lecture_meetings(c), c["code"]


def test_no_instructor_or_room_double_booked():
    for key in ("instructor_id", "room"):
        for _, group in itertools.groupby(sorted(current, key=lambda c: c[key]), key=lambda c: c[key]):
            assert not clashes(list(group)), key


def test_compulsory_courses_of_each_cohort_do_not_clash():
    cohorts = {
        "Semester III": [c for c in current if re.search(r"Semester III\b", c["offered_to"])],
        "Semester V": [c for c in current
                       if re.search(r"Semester V\b", c["offered_to"]) and "HSS-II" not in c["offered_to"]],
    }
    for name, group in cohorts.items():
        assert not clashes(group), name
    # A 7th-semester CSE student takes one core bouquet, Environmental Science and one Humanities III course.
    sem7_fixed = [c for c in current if c["category"] in {"core_bouquet", "institute_core"}]
    hum3 = [c for c in current if "Humanities III" in c["offered_to"]]
    for h in hum3:
        for b in sem7_fixed:
            assert not clashes([h, b]), (h["code"], b["code"])


def test_student_registration():
    profile = json.loads((STRUCTURED / "student_profile.json").read_text(encoding="utf-8"))
    by_code = {c["code"]: c for c in courses}
    registered = [by_code[code] for code in profile["registered_courses"]]
    assert all(c["semester"] == profile["academic_semester"] for c in registered)
    assert not clashes(registered)


def test_reference_answers():
    # "How many 400 level electives have been floated by the Humanities department this semester?"
    hss_400 = [c for c in current if c["level"] == "400" and "Humanities" in c["department"]]
    assert len(hss_400) == 7
    # "What is the slot for the course Autonomous Systems?"
    autonomous = next(c for c in courses if c["title"] == "Autonomous Systems")
    assert lecture_meetings(autonomous) == [("Tuesday", "11:00", "11:50"), ("Thursday", "11:00", "11:50"),
                                            ("Friday", "11:00", "11:50")]
    assert autonomous["room"] == "LHC 106"
    # "Write an email to the course instructor of CSL7570 ..."
    csl7570 = next(c for c in courses if c["code"] == "CSL7570")
    assert next(f for f in faculty if f["instructor_id"] == csl7570["instructor_id"])["email"].endswith("@iitj.ac.in")
