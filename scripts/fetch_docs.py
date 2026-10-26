"""Download the public IIT Jodhpur documents that form the helpdesk corpus.

PDFs are saved as-is; HTML pages are reduced to their main content and saved as
Markdown. A manifest (data/manifest.csv) records provenance for every document so
answers can cite the original source.

Usage:  uv run python scripts/fetch_docs.py [--force]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from datetime import date
from pathlib import Path

import pymupdf
import requests
import truststore
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.csv"

IITJ = "https://iitj.ac.in"
GALLERY = f"{IITJ}/PageImages/Gallery"
INTRA = "https://intranet.iitj.ac.in/acad_website"
ACAD = f"{IITJ}/office-of-academics/en"

# (doc_id, title, category, url)
SOURCES: list[tuple[str, str, str, str]] = [
    # --- Rules and regulations -------------------------------------------------
    ("reg_2019", "Academic Regulations for Students Enrolled from July 2019 Onwards", "regulation",
     f"{GALLERY}/03-2025/1_Academic_Regulations_Final_03_09_2019.pdf"),
    ("circ_medtech_joint_2022", "Regulations for Joint Programmes in Medical Technologies (Circular IITJ/OA/C/2021-22/21, 20 Jan 2022)", "circular",
     f"{GALLERY}/03-2025/5_Circular_IITJ_OA_C_2021-22_21_122_20.01.2022.pdf"),
    ("reg_pg_2022", "Regulations for PG Students Enrolled from July 2022 Onwards", "regulation",
     f"{GALLERY}/03-2025/4_Regulation_PG_2022-onwards_20022023.pdf"),
    ("reg_msr_2023", "M.S. by Research Regulations for Students Enrolled from July 2023 Onwards", "regulation",
     f"{GALLERY}/03-2025/6_2024-04-17-661f605b54457-1713332315.pdf"),
    ("amend_attendance_2024", "Amendment in the Attendance Policy of the Institute (Circular, 08 July 2024)", "circular",
     f"{GALLERY}/03-2025/7.1_Circular_51Circular-No.-IITJOAACircular2024-2551682-dated-08-July-2024.pdf"),
    ("amend_exam_2024", "Amendment in the Examination Policy of the Institute (Circular, 08 July 2024)", "circular",
     f"{GALLERY}/03-2025/7.2_Circular_52Circular-No.-IITJOAACircular2024-2552683-dated-08-July-2024.pdf"),
    ("amend_continuation_2020", "Revised Norms for Continuation of Program/Assistantship and Graduation Requirements (Notification, 26 Oct 2020)", "circular",
     f"{GALLERY}/03-2025/amend_notification_26102020.pdf"),
    # --- Calendars -------------------------------------------------------------
    ("calendar_2026_27", "Academic Calendar for Academic Year 2026-27", "calendar",
     f"{GALLERY}/06-2026/Academic-Calendar-AY-2026-27-639178076336417495.pdf"),
    ("calendar_summer_2026", "Academic Calendar for Summer Term 2026", "calendar",
     f"{GALLERY}/04-2026/Academic-Calendar-for-Summer-Term-2026--639124648616829491.pdf"),
    ("holidays_2026", "List of Institute Holidays 2026", "calendar",
     f"{GALLERY}/03-2026/List-of-holdays-for-2026-639080479940387076.pdf"),
    # --- Fees, scholarships, forms ---------------------------------------------
    ("fee_2026_27", "Fee Structure for AY 2026-27 (Regular Programs)", "office_order",
     f"{GALLERY}/07-2026/Fee-structure-for-AY-2026-27-Regular-639186884218584492.pdf"),
    ("scholarships", "Details of Scholarships/Fellowships/Fee Reimbursement for UG and PG Students", "office_order",
     f"{GALLERY}/03-2025/Details-of-UG-and-PG-Students-Scholarship-638780605862907546.pdf"),
    ("cgpa_conversion", "CGPA Conversion and Medium of Teaching Certificate", "office_order",
     f"{GALLERY}/04-2025/CGPA-Conversion-638797086253041021.pdf"),
    ("form_medical_certificate", "Medical Certificate Form", "form",
     f"{GALLERY}/07-2026/Medical-Certificate--639192976055448529.pdf"),
    # --- UG curricula (2019-2025 batches; a 7th-semester student in 2026-27 joined in 2023) ---
    ("btech_cse_structure", "B.Tech. (Computer Science & Engineering) Program Structure", "curriculum",
     f"{GALLERY}/03-2025/3%20B.Tech%20CSE%20Program%20structure_07102020.pdf"),
    ("btech_cse_curriculum", "B.Tech. (Computer Science & Engineering) Curriculum and Course Contents", "curriculum",
     f"{INTRA}/intra/3.%20B.Tech%20CSE_09102020.pdf"),
    ("btech_ee_structure", "B.Tech. (Electrical Engineering) Program Structure", "curriculum",
     f"{GALLERY}/03-2025/2%20B.Tech%20EE%20Program%20structure_07102020.pdf"),
    ("btech_me_structure", "B.Tech. (Mechanical Engineering) Program Structure", "curriculum",
     f"{GALLERY}/03-2025/1%20B.Tech%20ME%20Program%20structure_07102020.pdf"),
    ("btech_aids_structure", "B.Tech. (AI and Data Science) Program Structure", "curriculum",
     f"{GALLERY}/03-2025/5%20B.Tech%20AI%20&%20DS%20Program%20structure_16102020.pdf"),
    ("common_courses", "List of Institute-wide Common Courses and Course Contents from 2nd Year Onwards", "curriculum",
     f"{INTRA}/intra/List%20of%20Institute%20wide%20common%20courses%20and%20course%20contents%20from%202nd%20year%20onwards_31082020.pdf"),
    ("hss_courses", "Courses Offered by the Department of Humanities and Social Sciences (HSS)", "curriculum",
     f"{INTRA}/intra/Final_ConceptNote_HSS_2021%20with%20all%20modifications%20_13.01.2022.pdf"),
    ("non_graded_courses", "Non-Graded Institute Level Courses", "curriculum",
     f"{INTRA}/pg_curriculum/NON%20GRADED_06092019.pdf"),
    # --- PG curricula (CSL7xxx courses live here) -------------------------------
    ("mtech_cse_curriculum", "M.Tech. (Computer Science & Engineering) Curriculum", "curriculum",
     f"{GALLERY}/03-2025/MTech-CSE-curriculum.pdf"),
    ("mtech_ai_curriculum", "M.Tech. (Artificial Intelligence) Curriculum", "curriculum",
     f"{GALLERY}/03-2025/MTech-AI-curriculum.pdf"),
    # --- Web pages (main content saved as Markdown) ------------------------------
    ("web_ug_registration", "UG Registration Guidelines (HSS Bouquets)", "webpage", f"{ACAD}/ug-registration-guidelines"),
    ("web_specialization_minor", "Specialization / Minor / Dual Degree", "webpage", f"{ACAD}/specialization-minor-dual-degree"),
    ("web_engineering_science", "Engineering Science Programs", "webpage", f"{ACAD}/engineering-science"),
    ("web_academic_programs", "List of Academic Programs", "webpage", f"{ACAD}/list-of-academic-programs"),
    ("web_academics_contact", "Office of Academic Affairs - Contact", "webpage", f"{ACAD}/contact"),
]

# iitj.ac.in serves an incomplete certificate chain; the OS trust store resolves it, certifi does not.
truststore.inject_into_ssl()

HEADERS = {"User-Agent": "Mozilla/5.0 (IITJ academic-helpdesk coursework project)"}


def html_to_markdown(html: str, title: str, url: str) -> str:
    """Keep only the page body between the sub-page title and the site footer."""
    start = html.find('class="subpage-title"')
    end = html.find("footer-bg", start)
    if start != -1 and end != -1:
        html = html[html.rfind("<", 0, start): html.rfind("<", 0, end)]
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            width = rows[0].count("|") - 1
            rows.insert(1, "|" + " --- |" * width)
        table.replace_with("\n" + "\n".join(rows) + "\n")

    text = soup.get_text("\n")
    lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in text.splitlines()]
    body = "\n".join(ln for ln in lines if ln)
    return f"# {title}\n\nSource: {url}\n\n{body}\n"


def fetch(doc_id: str, title: str, url: str, force: bool) -> Path:
    is_web = not (url.lower().endswith(".pdf") or "drive.google.com" in url)
    out = RAW_DIR / f"{doc_id}.{'md' if is_web else 'pdf'}"
    if out.exists() and not force:
        return out

    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    if is_web:
        out.write_text(html_to_markdown(resp.text, title, url), encoding="utf-8")
    else:
        if not resp.content.startswith(b"%PDF"):
            raise ValueError(f"not a PDF (content-type {resp.headers.get('Content-Type')})")
        out.write_bytes(resp.content)
    return out


def page_count(path: Path) -> int | str:
    if path.suffix != ".pdf":
        return ""
    with pymupdf.open(path) as doc:
        return doc.page_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for doc_id, title, category, url in SOURCES:
        try:
            path = fetch(doc_id, title, url, args.force)
        except Exception as exc:  # keep going; report at the end
            failures.append((doc_id, url, exc))
            print(f"FAIL {doc_id}: {exc}", file=sys.stderr)
            continue
        rows.append({
            "doc_id": doc_id,
            "title": title,
            "category": category,
            "file": path.relative_to(ROOT).as_posix(),
            "source_url": url,
            "fetched_on": date.today().isoformat(),
            "pages": page_count(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        })
        print(f"ok   {doc_id:<28} {path.name}")

    if not rows:
        return 1
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} documents saved, {len(failures)} failed -> {MANIFEST.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
