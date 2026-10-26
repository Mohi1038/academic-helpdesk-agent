# Structured data (Semester I, AY 2026-27)

These CSVs back the lookup tools (course and slot lookup, course counting, instructor contact). The
target user is a **7th-semester B.Tech. student in AY 2026-27 Semester I** (`2026-27-I`), who joined in 2023
and therefore follows the 2019–2025 curriculum.

| File | Contents |
| --- | --- |
| `courses.csv` | Courses floated in `2026-27-I`, plus a few `2025-26-II` rows so that semester filtering is meaningful |
| `timetable.csv` | Slot system for B.Tech. 2nd year onwards: lecture slots A, C, F, G, I, K and N; lab slots L1–L10 and M1–M10 |
| `faculty.csv` | Instructor directory |
| `student_profile.json` | The user's own registration: the six courses on their timetable (name and roll number are placeholders) |

`courses.csv` describes everything the institute floats. The student is registered only in the six courses
listed in `student_profile.json`. Questions about "my classes" or "my instructor" resolve through the profile;
institute-wide questions, such as counting Humanities electives, use the full course list.

## Timetable

`timetable.csv` is transcribed from the official *Slot System for B.Tech. 2nd year onwards (AY 2026-27:
Semester I)* sheet.

That sheet shows six courses by name in place of a slot letter, and letters B, D, E, H and J never appear in it.
Rather than guess letters, those six courses have an empty `slot` and list their actual lecture times in
`meetings` instead:

| Course | Meetings | Room |
| --- | --- | --- |
| Artificial Intelligence | Tue 10:00, Wed 09:00, Fri 09:00 | LHC 110 |
| Autonomous Systems | Tue, Thu, Fri 11:00 | LHC 106 |
| Operations Research for Economics | Mon 11:00, Wed 11:00, Fri 10:00 | LHC 206 |
| Introduction to Business and Management | Mon, Wed, Fri 12:00 | LHC 308 |
| Environmental Science | Tue, Thu 12:00 | LHC 110 |
| Computational Game Theory | Tue, Thu, Fri 16:00 | LHC 204 |

Every other course has a slot letter, and its times come from `timetable.csv`.

## What is real and what is not

| Field | Source |
| --- | --- |
| Course titles, levels, L-T-P, credits | Real: from the IITJ curriculum documents named in `title_source` (`doc_id` in `data/manifest.csv`), or the student timetable sheet |
| Course codes | Real where the HSS document publishes them (`HSL3060`, `HSL3070`, `HSL4010`, `HSL4020`, `HSL4030`, `HSL4070`, `HSL7460`, `AIL4010`). All others are **synthetic**, following the IITJ pattern `<dept><L/P><level digit><3 digits>` |
| Times and rooms of the six named courses | Real: from the timetable sheet |
| Slots and rooms of all other courses | **Synthetic.** Chosen so that no instructor, room or student cohort is double-booked |
| Faculty names, designations, emails, research areas | Real: from the public department pages at `iitj.ac.in/People/List?dept=…` (fetched 2026-10-26) |
| Course-to-instructor assignment | **Illustrative.** No public source gives instructor allocations, so each course is matched to a faculty member of the relevant department whose research area fits it. These are not official teaching assignments |

The humanities courses are listed under *School of Liberal Arts (Humanities and Social Sciences)*. The 2021 HSS
course document uses the older department name, and the current site lists the School of Liberal Arts.

Facts the evaluation relies on (also checked by `tests/test_data.py`):

- 7 courses at the 400 level are floated by the humanities department in `2026-27-I`. `AIL4010` is cross-listed in the humanities bouquet but belongs to SAIDE, and is not counted.
- Autonomous Systems (`IDL4010`) meets Tue, Thu and Fri 11:00–11:50 in LHC 106.
- CSL7570 (Natural Language Understanding) is assigned to Sidharth Ranjan (`sranjan@iitj.ac.in`).
