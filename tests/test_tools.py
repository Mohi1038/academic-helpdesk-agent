"""Tool behaviour on the sample questions from the assignment, plus edge cases."""

import json

import pytest

from helpdesk import config, rag
from helpdesk.tools import (
    draft_email, find_course, get_instructor_contact, get_my_schedule, get_slot_timetable,
    list_courses, log_unanswered, search_documents,
)


@pytest.fixture(autouse=True)
def temp_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DRAFTS_DIR", tmp_path / "drafts")
    monkeypatch.setattr(config, "UNANSWERED_LOG", tmp_path / "logs" / "unanswered.jsonl")


def test_count_400_level_humanities_electives():
    for wording in ("Humanities", "HSS", "humanities department", "School of Liberal Arts"):
        result = list_courses(department=wording, level=400, category="electives")
        assert result["count"] == 7, wording
        assert all(c["code"].startswith("HSL4") for c in result["courses"])


def test_list_courses_unknown_department_lists_valid_ones():
    result = list_courses(department="Astrology")
    assert result["status"] == "unknown_department"
    assert "Economics" in result["departments"]


def test_find_course_by_title_and_messy_code():
    autonomous = find_course("autonomous systems")["courses"][0]
    assert autonomous["lectures"] == "Tuesday 11:00-11:50, Thursday 11:00-11:50, Friday 11:00-11:50"
    assert autonomous["room"] == "LHC 106"
    assert find_course("csl 7570")["courses"][0]["title"] == "Natural Language Understanding"
    assert find_course("Quantum Gastronomy")["status"] == "not_found"


def test_find_course_with_slot_letter_and_lab():
    dsa = find_course("CSL2020")["courses"][0]
    assert dsa["slot"] == "A"
    assert dsa["lab"].startswith("slot M6: Monday 15:00-16:50")


def test_slot_lookup():
    assert len(get_slot_timetable("slot a")["meetings"]) == 3
    assert get_slot_timetable("Z")["status"] == "not_found"


def test_my_schedule_only_has_registered_courses():
    tuesday = get_my_schedule("Tue")
    assert [c["title"] for c in tuesday["classes"]] == [
        "Artificial Intelligence", "Autonomous Systems", "Environmental Science", "Computational Game Theory"]
    assert len(get_my_schedule()["registered_courses"]) == 6
    assert get_my_schedule("Funday")["status"] == "error"


def test_instructor_contact_for_csl7570():
    info = get_instructor_contact("CSL7570")["instructors"][0]
    assert info["email"] == "sranjan@iitj.ac.in"


def test_draft_email_saved_with_signature_and_not_sent():
    result = draft_email("sranjan@iitj.ac.in", "Leave application for 2 days",
                         "Dear Prof. Ranjan,\n\nI request leave for two days.")
    assert result["status"] == "draft_saved"
    assert "Roll No.:" in result["email"] and result["email"].startswith("To: Sidharth Ranjan")
    assert open(result["draft_file"], encoding="utf-8").read() == result["email"]


@pytest.mark.parametrize("closing", [
    "\n\nSincerely,",
    "\n\nBest regards,\n[Your Name] [Your Roll Number]",
    "\n\nYours sincerely,\nA Student\nB23CS000",
])
def test_draft_email_strips_model_sign_off(closing):
    body = "Dear Prof. Jain,\n\nI will miss Thursday's class due to fever.\n\nThank you for your understanding."
    email = draft_email("anoopj@iitj.ac.in", "Absence", body + closing)["email"]
    assert email.count("Regards,") == 1 and "[Your Name]" not in email and "Sincerely" not in email
    assert "Thank you for your understanding." in email  # a closing sentence inside the message stays


def test_draft_email_rejects_addresses_not_in_directory():
    assert draft_email("someone@gmail.com", "Hi", "Dear Sir,")["status"] == "error"


def test_log_unanswered_appends_jsonl():
    log_unanswered("Where can I eat on campus?", "out of scope: dining")
    log_unanswered("Wi-Fi password?", "out of scope")
    lines = config.UNANSWERED_LOG.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["question"] for line in lines] == ["Where can I eat on campus?", "Wi-Fi password?"]


class FakeRetriever:
    def __init__(self, similarity):
        self.similarity = similarity

    def search(self, query, k=5):
        return [rag.Hit("reg::001", "reg", "Academic Regulations", "url", 7, 8, "", "text " * 2000,
                        self.similarity, 0.03)]

    def best_similarity(self, hits):
        return max(h.similarity for h in hits)


def test_search_documents_returns_cited_truncated_passages(monkeypatch):
    monkeypatch.setattr("helpdesk.tools.documents.retriever", FakeRetriever(0.8))
    result = search_documents("attendance")
    assert result["results"][0]["citation"] == "Academic Regulations, pp. 7-8"
    assert len(result["results"][0]["text"]) == config.MAX_CHUNK_CHARS


def test_search_documents_refuses_below_threshold(monkeypatch):
    monkeypatch.setattr("helpdesk.tools.documents.retriever", FakeRetriever(0.3))
    assert search_documents("dining options")["status"] == "no_relevant_documents"
