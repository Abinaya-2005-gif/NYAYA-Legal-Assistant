import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)


# ---------- Configuration ----------

MODEL = os.getenv("NYAYA_MODEL", "nova-micro")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = str(BASE_DIR / "db")

DOMAINS = [
    "childrens_rights",
    "consumer_rights",
    "criminal_rights",
    "cyber_rights",
    "fundamental_rights",
    "labour_rights",
    "legal_rights",
    "rti",
    "women_rights",
    "other"
]

DISCLAIMER = (
    "\n\n*This is general legal information, not legal advice. "
    "Please consult a lawyer or your District Legal Services Authority (DLSA).*"
)

HELPLINES = (
    "\n\n**Helplines:** Free legal aid (NALSA) 15100 | "
    "Women helpline 181 | Police 112 | Consumer helpline 1915"
)


# ---------- Disclaimer Cleanup ----------

def clean_model_answer(answer):

    if not answer:
        return ""

    answer = answer.strip()

    paragraphs = re.split(r"\n\s*\n", answer)

    cleaned_paragraphs = []

    for paragraph in paragraphs:

        lower = paragraph.lower()

        if (
            "this is general legal information" in lower
            or "not professional legal advice" in lower
            or "not legal advice" in lower
            or "please consult a lawyer" in lower
            or "district legal services authority (dlsa)" in lower
        ):
            continue

        cleaned_paragraphs.append(paragraph.strip())

    return "\n\n".join(cleaned_paragraphs).strip()


# ---------- Resources ----------

@lru_cache(maxsize=1)
def _embedder():

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "intfloat/multilingual-e5-small"
    )


def _tok(text):

    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=1)
def _kb():

    import chromadb
    from rank_bm25 import BM25Okapi

    col = chromadb.PersistentClient(
        path=DB_PATH
    ).get_or_create_collection("laws")

    data = col.get()

    if not data["ids"]:

        raise RuntimeError(
            "Knowledge base is empty. Run: python ingest.py"
        )

    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    return {
        "col": col,
        "ids": ids,
        "docs": dict(zip(ids, documents)),
        "metas": dict(zip(ids, metadatas)),
        "bm25": BM25Okapi(
            [_tok(d) for d in documents]
        )
    }


# ---------- Nexus API ----------

def ask(system, user, max_tokens=1200):

    key = os.getenv("NEXUS_API_KEY")
    base_url = os.getenv("NEXUS_API_BASE_URL")

    if not key:

        raise RuntimeError(
            "NEXUS_API_KEY missing in .env"
        )

    if not base_url:

        raise RuntimeError(
            "NEXUS_API_BASE_URL missing in .env"
        )

    url = base_url.rstrip("/") + "/chat/completions"

    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": user
            }
        ],
        "max_tokens": max_tokens * 3
    }

    for attempt in range(5):

        try:

            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json=body,
                timeout=120
            )

            if response.status_code in (
                429, 500, 502, 503, 504
            ):

                time.sleep(5 * (attempt + 1))
                continue

            if response.status_code != 200:

                raise RuntimeError(
                    f"Nexus API Error "
                    f"{response.status_code}: {response.text}"
                )

            data = response.json()

            return data["choices"][0]["message"]["content"]

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError
        ):

            time.sleep(5 * (attempt + 1))
            continue

    raise RuntimeError(
        "Nexus API is busy or slow. Please try again."
    )


# ---------- JSON Helper ----------

def ask_json(system, user, default=None, retries=2):

    for _ in range(retries + 1):

        try:

            raw = ask(
                system + "\nRespond with ONLY valid JSON, nothing else.",
                user,
                600
            )

            start = raw.find("{")

            if start == -1:
                continue

            decoder = json.JSONDecoder()

            result, _ = decoder.raw_decode(
                raw[start:]
            )

            if isinstance(result, dict):
                return result

        except (
            ValueError,
            RuntimeError,
            KeyError,
            TypeError
        ):

            continue

    return default if default is not None else {}


# ---------- Agent 1: Analyzer ----------

def analyze(question):

    res = ask_json(

        "You are an Indian legal domain classification assistant.",

        f"""
Analyze this legal question.

Question:
{question}

Return JSON with these keys:

"language": English, Tamil, Hindi or other.
Recognize Tamil and Hindi even when typed using English letters.

"domain": Choose exactly ONE from:

{DOMAINS}

Domain classification:

- childrens_rights: Child protection, POCSO, child abuse.
- consumer_rights: Consumer complaints, defective products, refunds.
- criminal_rights: BNS, BNSS, BSA, theft, murder, assault.
- cyber_rights: Cybercrime, online fraud, hacking.
- fundamental_rights: Constitution and fundamental rights.
- labour_rights: Unpaid salary, delayed wages, employer not paying salary, minimum wages, wage deductions, employment and labour disputes.
- legal_rights: General Indian laws and legal information.
- rti: Right to Information.
- women_rights: Domestic violence, women's safety, women's rights.
- other: If none of the above matches.

IMPORTANT:
Questions about unpaid salary, pending salary,
delayed salary or wage disputes MUST be classified
as labour_rights.

"english_query":
Rewrite the question as a clear English legal search query.

"urgent":
true if the user describes immediate danger to life or safety.
Otherwise false.

Return valid JSON only.
""",

        default={
            "language": "English",
            "domain": "other",
            "english_query": question,
            "urgent": False
        }
    )

    res.setdefault("language", "English")
    res.setdefault("domain", "other")
    res.setdefault("english_query", question)
    res.setdefault("urgent", False)

    domain_aliases = {
        "children": "childrens_rights",
        "child": "childrens_rights",
        "consumer": "consumer_rights",
        "criminal": "criminal_rights",
        "cyber": "cyber_rights",
        "fundamental": "fundamental_rights",
        "labour": "labour_rights",
        "labor": "labour_rights",
        "employment": "labour_rights",
        "wages": "labour_rights",
        "salary": "labour_rights",
        "women": "women_rights",
        "general": "legal_rights"
    }

    domain = str(
        res["domain"]
    ).lower().strip()

    res["domain"] = domain_aliases.get(
        domain,
        domain
    )

    if res["domain"] not in DOMAINS:
        res["domain"] = "other"

    if not isinstance(res["urgent"], bool):
        res["urgent"] = False

    return res

# ---------- Emergency Detection ----------

def detect_urgent(question):

    text = question.lower()

    urgent_phrases = [

        "immediate danger",
        "in immediate danger",
        "danger right now",
        "unsafe right now",
        "being attacked",
        "attacking me",
        "threatening to kill",
        "trying to kill me",
        "physical attack",
        "life threatening",
        "help me immediately",
        "currently being abused",
        "violence happening now",
        "afraid for my life",

        "ippo adikkiranga",
        "ennai adikkiranga",
        "kolven",
        "kolrathaga mirattal",
        "uyirukku aabathu",
        "ippo danger",
        "ippo unsafe"
    ]

    return any(
        phrase in text
        for phrase in urgent_phrases
    )


# ---------- Agent 2: Hybrid Retriever ----------
def retrieve(query, domain=None, k=10):

    kb = _kb()

    ids = kb["ids"]
    metas = kb["metas"]

    # ---------- Domain Filtering ----------

    pool = [
        i for i in ids
        if domain in (None, "other")
        or metas[i].get("domain") == domain
    ]

    if not pool:
        return []

    pool_set = set(pool)

    # ---------- Query Expansion ----------

    queries = [query]

    q_lower = query.lower()

    if "domestic violence" in q_lower:

        queries.append(
            "Section 3 definition of domestic violence "
            "physical abuse sexual abuse verbal emotional "
            "abuse and economic abuse"
        )

        queries.append(
            "meaning and explanation of domestic violence "
            "under Protection of Women from Domestic Violence Act 2005"
        )

    if "protection order" in q_lower:

        queries.append(
            "Section 18 protection orders Magistrate "
            "prohibit domestic violence"
        )

        queries.append(
            "Section 12 application to Magistrate "
            "seeking relief protection order"
        )

    # ---------- Reciprocal Rank Fusion ----------

    fused = {}

    for search_query in queries:

        # Vector Search

        try:

            q_emb = _embedder().encode(
                ["query: " + search_query],
                normalize_embeddings=True
            ).tolist()

            if domain not in (None, "other"):

                result = kb["col"].query(
                    query_embeddings=q_emb,
                    n_results=min(k * 5, len(pool)),
                    where={"domain": domain}
                )

            else:

                result = kb["col"].query(
                    query_embeddings=q_emb,
                    n_results=min(k * 5, len(ids))
                )

            vec_ids = [
                doc_id
                for doc_id in result["ids"][0]
                if doc_id in pool_set
            ]

        except Exception:

            vec_ids = []

        # BM25 Search

        scores = kb["bm25"].get_scores(
            _tok(search_query)
        )

        pool_indices = [
            index
            for index, doc_id in enumerate(ids)
            if doc_id in pool_set
        ]

        bm_indices = sorted(
            pool_indices,
            key=lambda index: scores[index],
            reverse=True
        )[:min(k * 5, len(pool_indices))]

        bm_ids = [
            ids[index]
            for index in bm_indices
        ]

        # ---------- Combine Rankings ----------

        for rank, doc_id in enumerate(vec_ids):

            fused[doc_id] = (
                fused.get(doc_id, 0)
                + 1 / (60 + rank + 1)
            )

        for rank, doc_id in enumerate(bm_ids):

            fused[doc_id] = (
                fused.get(doc_id, 0)
                + 1 / (60 + rank + 1)
            )

    # ---------- Final Ranking ----------

    best = sorted(
        fused,
        key=fused.get,
        reverse=True
    )[:k]

    return [
        {
            "id": doc_id,
            "text": kb["docs"][doc_id],
            "meta": kb["metas"][doc_id]
        }
        for doc_id in best
    ]

# ---------- Format Legal Context ----------

def format_context(docs):

    if not docs:
        return "No legal documents retrieved."

    return "\n\n".join(

        f"[{d['meta'].get('act', 'Unknown Act')} - "
        f"{d['meta'].get('section', 'Unknown Section')}]\n"
        f"{d['text']}"

        for d in docs
    )


# ---------- Agent 3: Grader ----------

def grade(question, docs):

    if not docs:
        return False

    res = ask_json(

        "You judge whether retrieved legal text can answer a citizen's question.",

        f"""
Question:

{question}

Retrieved legal text:

{format_context(docs)}

Check whether the retrieved text is directly relevant
to the legal question.

Return JSON:

{{
    "relevant": true or false,
    "reason": "short reason"
}}
""",

        default={
            "relevant": False,
            "reason": "Grading could not be completed."
        }
    )

    return res.get("relevant") is True


# ---------- Agent 4: Rewriter ----------

def rewrite(question, old_query):

    return ask(

        "You rewrite search queries to find Indian law provisions.",

        f"""
Original question:

{question}

Previous query that failed:

{old_query}

Write ONE improved English search query using
relevant Indian legal terminology.

Output only the query.
""",

        100
    ).strip()


# ---------- Agent 5: Generator ----------

def generate(
    question,
    docs,
    language,
    conversation_history=""
):

    system = f"""
You are NYAYA, an Indian legal information assistant.

STRICT RULES:

1. Answer ONLY using the provided legal context.
2. If context is insufficient, clearly mention it.
3. Use simple, clear and natural language.
4. Reply entirely in {language}.
5. Understand the user's original question correctly.
6. NEVER change the meaning or intent of the user's question.
7. Answer every part of a multi-part question.
8. Explain legal definitions when supported by the context.
9. Explain practical next steps only when supported by
   the available legal sources.
10. Cite the exact Act name and section number.
11. Never invent Acts, sections or legal procedures.
12. Never predict case outcomes.
13. Use previous conversation details when relevant.
14. Do not repeat questions already answered.
15. Do not include a legal disclaimer.
    The application adds it automatically.
16. Do not present unsupported legal claims as facts.
17. Never guess a section number.
18. Every section number must belong to the exact Act
    mentioned in the citation.
19. Never mix section numbers from different Acts.
20. Use only Act names and section details available
    in the retrieved legal context.
21. If a legal provision cannot be confirmed from
    the retrieved context, clearly state that it
    could not be verified.

TAMIL LANGUAGE RULES:

22. Use natural, grammatically correct Tamil.
23. Do not translate legal terms word by word.
24. Use appropriate Tamil legal terminology.
25. For example:
    - Domestic Violence: குடும்ப வன்முறை
    - Protection Order: பாதுகாப்பு ஆணை
    - Magistrate: மாஜிஸ்திரேட் (நீதித்துறை நடுவர்)
    - Aggrieved Person: பாதிக்கப்பட்ட பெண்
26. Never translate domestic violence as political violence.
27. Preserve the original meaning of the user's question.
28. Avoid unnecessary repetition and confusing words.

ANSWER STRUCTURE:

- Directly answer the user's question.
- Explain relevant legal provisions.
- Give practical next steps if supported by sources.
- Clearly mention missing information if the retrieved
  sources do not cover part of the question.
"""

    user_prompt = f"""
Previous Conversation:

{conversation_history}

Current User Question:

{question}

Legal Context:

{format_context(docs)}

Answer every part of the question using the available
legal sources.

Do not add unsupported information.
"""

    answer = ask(
        system,
        user_prompt,
        1500
    )

    return clean_model_answer(answer)


# ---------- Agent 6: Verifier ----------

def verify(answer, docs):

    if not docs:
        return {
            "supported": False,
            "issues": "No legal sources available."
        }

    # ---------- Hard-coded Citation Check ----------

    wrong_dv_citation = re.search(
        r"section\s*2\s*\(\s*g\s*\)",
        answer,
        re.IGNORECASE
    )

    dv_definition_claim = re.search(
        r"(definition of domestic violence|"
        r"defines domestic violence|"
        r"domestic violence is defined|"
        r"domestic violence under section)",
        answer,
        re.IGNORECASE
    )

    if wrong_dv_citation and dv_definition_claim:
        return {
            "supported": False,
            "issues": (
                "Incorrect legal citation: Section 2(g) "
                "defines domestic relationship, not domestic "
                "violence. Section 3 of the Protection of "
                "Women from Domestic Violence Act, 2005 "
                "contains the definition of domestic violence."
            )
        }

    # ---------- AI Verification ----------

    context = format_context(docs)

    result = ask_json(
        """
You are a strict Indian legal citation verification agent.

Verify every important legal claim using ONLY
the retrieved legal documents.

Rules:
1. Check every Act name and section number.
2. Check whether the section heading and content
   actually support the claim.
3. Do not approve a citation just because its
   section number appears in the documents.
4. Reject incorrect definitions, procedures,
   rights and legal claims.
5. If sources are insufficient, return false.
6. If uncertain, return false.

Return ONLY valid JSON:
{
    "supported": true,
    "issues": "All important claims are supported."
}
""",
        f"""
RETRIEVED LEGAL SOURCES:

{context}

GENERATED ANSWER:

{answer}
""",
        default={
            "supported": False,
            "issues": "Verification failed."
        }
    )

    supported = (
        isinstance(result, dict)
        and result.get("supported") is True
    )

    return {
        "supported": supported,
        "issues": result.get(
            "issues",
            "Verification incomplete."
        )
    }



# ---------- Agent 7: Fallback ----------

def fallback(question, language):

    system = f"""
You are NYAYA, a legal information assistant
for common Indian citizens.

The knowledge base had NO relevant verified source.

Rules:

1. Start by warning in {language} that this topic
   is not covered by the verified database.
2. Clearly state that the answer is not source-verified.
3. Name the likely relevant Indian law only
   if you are confident.
4. NEVER invent section numbers.
5. Use very simple language.
6. Reply in {language}.
7. Give practical next steps where appropriate.
8. Mention police, consumer forum, DLSA,
   NALSA 15100 or a lawyer when relevant.
9. Never predict case outcomes.
10. Do not include a legal disclaimer.
    The application adds it automatically.
"""

    answer = ask(
        system,
        question,
        1200
    )

    return clean_model_answer(answer)


# ---------- Baseline: Plain LLM ----------

def plain_llm(
    question,
    language="the same language as the question"
):

    answer = ask(
        f"""
You are a legal information assistant for Indian citizens.

Reply in {language}.
Use simple language.
Do not include a legal disclaimer.
""",
        question,
        1200
    )

    return clean_model_answer(answer)


# ---------- Follow-up Question Agent ----------

def check_clarification(
    question,
    conversation_history=""
):

    q = question.lower()

    general_legal_questions = [
        "domestic violence-na enna",
        "what is domestic violence",
        "meaning of domestic violence",
        "protection order eppadi",
        "how to get a protection order",
        "how can i get a protection order",
        "what is a protection order"
    ]

    if any(
        term in q
        for term in general_legal_questions
    ):

        return {
            "needs_clarification": False,
            "questions": []
        }

    system = """
You are NYAYA's legal clarification agent.

Your job is to identify whether essential information
is missing from the user's legal question.

STRICT RULES:

1. Ask clarification questions only when essential details
   are missing to understand the user's legal situation.

2. Ask a maximum of 2 questions at a time.

3. Never repeat questions already asked or answered
   in the previous conversation.

4. Do not ask unnecessary personal or background questions.

5. For direct legal questions about a specific Act,
   Section, legal definition, or procedure, do not ask
   follow-up questions.

6. If the user has already provided sufficient information,
   proceed directly to legal research.

7. If the user does not know an answer, do not keep
   asking the same question.

8. Ask clarification only once for a particular query.
   If previous conversation already contains clarification
   questions, set needs_clarification to false.

9. Never create an endless clarification loop.

10. If immediate danger is described, do not delay
    urgent safety guidance with unnecessary questions.

Return ONLY valid JSON:

{
    "needs_clarification": true,
    "questions": ["Question 1"]
}

If clarification is not necessary:

{
    "needs_clarification": false,
    "questions": []
}
"""

    user_prompt = f"""
Previous Conversation:

{conversation_history}

Current User Question:

{question}

Determine whether essential information is missing.
Check previous questions carefully to avoid repetition.
"""

    return ask_json(
        system,
        user_prompt,
        default={
            "needs_clarification": False,
            "questions": []
        }
    )


# ---------- Main Pipeline ----------

def run(
    question,
    mode="agentic",
    max_retries=2,
    forced_language=None,
    conversation_history=""
):

    trace = []

    # Step 1: Clarification

    clarification = check_clarification(
        question,
        conversation_history
    )

    if clarification.get("needs_clarification", False):

        questions = clarification.get("questions", [])

        answer = (
            "To understand your situation better, "
            "please answer these questions:\n\n"
        )

        for i, q in enumerate(questions, 1):
            answer += f"{i}. {q}\n"

        trace.append(
            "Clarification Agent -> More information required"
        )

        return {
            "answer": answer,
            "docs": [],
            "trace": trace,
            "mode": "Follow-up Questions",
            "relevant": None,
            "supported": None
        }

    # Step 2: Analyzer

    full_situation = f"""
Previous Conversation:

{conversation_history}

Current Question:

{question}
"""

    info = analyze(full_situation)

    language = forced_language or info["language"]

    urgent = (
        info["urgent"]
        or detect_urgent(full_situation)
    )

    query = info["english_query"]

    trace.append(
        f"Analyzer -> language={language}, "
        f"domain={info['domain']}, "
        f"urgent={urgent}"
    )

    # Step 3: Plain LLM

    if mode == "plain":

        answer = plain_llm(
            full_situation,
            language
        )

        if urgent:
            answer += HELPLINES

        return {
            "answer": clean_model_answer(answer) + DISCLAIMER,
            "docs": [],
            "trace": trace + [
                "Plain LLM (no retrieval)"
            ],
            "mode": "Plain LLM",
            "relevant": None,
            "supported": None
        }

    # Step 4: Basic RAG and Agentic RAG

    docs = []
    relevant = False
    supported = None

    if mode == "basic":

        docs = retrieve(
            query,
            domain=info["domain"]
        )

        trace.append(
            f"Basic RAG -> Retrieved {len(docs)} chunks"
        )

        relevant = grade(
            full_situation,
            docs
        )

        trace.append(
            f"Grader -> relevant={relevant}"
        )

        if relevant:

            answer = generate(
                question,
                docs,
                language,
                conversation_history
            )

            check = verify(
                answer,
                docs
            )

            supported = check["supported"]

            trace.append(
                f"Verifier -> supported={supported}"
            )

            if not supported:

                answer += (
                    "\n\n⚠️ Verification warning: "
                    "Some legal claims could not be "
                    "confirmed from the retrieved sources. "
                    + check["issues"]
                )

        else:

            answer = fallback(
                full_situation,
                language
            )

            docs = []

            trace.append(
                "Basic RAG -> No relevant source, fallback used"
            )

        mode_label = "Basic RAG"

    else:

        # ---------- Agentic RAG ----------

        for attempt in range(max_retries + 1):

            domain = (
                info["domain"]
                if attempt == 0
                else None
            )

            docs = retrieve(
                query,
                domain=domain
            )

            relevant = grade(
                full_situation,
                docs
            )

            trace.append(
                f"Retriever (try {attempt + 1}) "
                f"domain={domain or 'all'} "
                f"-> relevant={relevant}"
            )

            if relevant:
                break

            if attempt < max_retries:

                query = rewrite(
                    full_situation,
                    query
                )

                trace.append(
                    f"Rewriter -> {query}"
                )

        if relevant:

            answer = generate(
                question,
                docs,
                language,
                conversation_history
            )

            check = verify(
                answer,
                docs
            )

            supported = check["supported"]

            trace.append(
                f"Verifier -> supported={supported}"
            )

            if not supported:

                answer += (
                    "\n\n⚠️ Verification warning: "
                    "Some legal claims could not be "
                    "confirmed from the retrieved sources. "
                    + check["issues"]
                )

            mode_label = "RAG (source-backed)"

        else:

            answer = fallback(
                full_situation,
                language
            )

            docs = []

            trace.append(
                "Grader rejected all retries -> Fallback agent"
            )

            mode_label = (
                "Fallback (general knowledge, NOT verified)"
            )

    # Step 5: Urgent Situation Handling

    if urgent:

        answer += HELPLINES

        trace.append(
            "Urgent situation -> Helplines appended"
        )

    # Step 6: Final Response

    return {
        "answer": clean_model_answer(answer) + DISCLAIMER,
        "docs": docs,
        "trace": trace,
        "mode": mode_label,
        "relevant": relevant,
        "supported": supported
    }