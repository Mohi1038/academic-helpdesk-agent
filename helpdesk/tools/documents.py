from helpdesk import config
from helpdesk.rag import retriever


def search_documents(query: str) -> dict:
    """Searches IIT Jodhpur's official academic documents: rules and regulations, attendance and
    examination policies, circulars, the academic calendar, fees, scholarships, forms and curricula
    (programme structures, credit requirements, course contents, elective lists).

    Args:
        query: A self-contained search query, e.g. "attendance requirement for major examination".

    Returns:
        dict: `status` is "success" with `results` (citation, url, text), or
        "no_relevant_documents" when nothing in the corpus matches the query.
    """
    hits = retriever.search(query)
    if retriever.best_similarity(hits) < config.MIN_SIMILARITY:
        return {
            "status": "no_relevant_documents",
            "message": "The academic documents do not cover this question. Do not guess; tell the "
                       "user you do not know and call log_unanswered.",
        }
    return {
        "status": "success",
        # No passage ids: given "S1".."S5", small models cite the id instead of the document.
        "results": [
            {"citation": hit.citation, "url": hit.source_url, "text": hit.text[:config.MAX_CHUNK_CHARS]}
            for hit in hits
        ],
        "instructions": "Answer only from these passages. Cite each fact with the passage's citation text in "
                        "square brackets, e.g. [" + hits[0].citation + "]. If the passages do not contain "
                        "the answer, call log_unanswered and say you do not know.",
    }
