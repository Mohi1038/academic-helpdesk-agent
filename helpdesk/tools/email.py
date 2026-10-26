import re
from datetime import datetime

from helpdesk import catalog, config

CLOSING = re.compile(r"^\s*(yours (sincerely|truly|faithfully)|sincerely|(best |kind |warm )?regards|thanks|"
                     r"thank you)\s*[,.!]?\s*$", re.IGNORECASE)


def _strip_sign_off(body: str) -> str:
    """Cut a closing the model added anyway ("Best regards,") and any name/placeholder lines after it."""
    lines = body.strip().splitlines()
    for i in range(len(lines) - 1, max(len(lines) - 5, 0) - 1, -1):
        if CLOSING.match(lines[i]):
            return "\n".join(lines[:i]).rstrip()
    return body.strip()


def draft_email(to_email: str, subject: str, body: str) -> dict:
    """Saves an email draft to an instructor for the student to review. The email is NOT sent.
    Look up the recipient with get_instructor_contact first. Write the greeting and message in
    `body`, but no sign-off: the student's name and roll number are appended automatically.

    Args:
        to_email: Instructor's email address, exactly as returned by get_instructor_contact.
        subject: Email subject line.
        body: Email text starting with the greeting, e.g. "Dear Prof. ...,".

    Returns:
        dict: `status` "draft_saved" with the full `email` text and the draft file path, or "error".
    """
    directory = {f["email"].lower(): f for f in catalog.faculty().values()}
    recipient = directory.get(to_email.strip().lower())
    if recipient is None:
        return {"status": "error",
                "message": f"'{to_email}' is not in the faculty directory. Call get_instructor_contact "
                           "and use the email address it returns."}

    body = _strip_sign_off(body)

    profile = catalog.profile()
    signature = f"Regards,\n{profile['name']}\nRoll No.: {profile['roll_number']}\n{profile['programme']}"
    email = (f"To: {recipient['name']} <{recipient['email']}>\nSubject: {subject.strip()}\n\n"
             f"{body.strip()}\n\n{signature}\n")

    config.DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", subject.lower()).strip("-")[:40] or "draft"
    path = config.DRAFTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.txt"
    path.write_text(email, encoding="utf-8")
    return {"status": "draft_saved", "email": email, "draft_file": str(path),
            "note": "Draft only - it has not been sent. Show it to the student to review and send."}
