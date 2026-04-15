"""
Fine-Tuneing/prepare_finetune_data.py

Prepares training data for local Qwen3 4B LoRA finetuning.

Redesigns applied:
  1. Instruction-following preservation samples (~50 general Q&A)
     Prevents LoRA from eroding the base model's instruction-following.
  3. System prompt restored to training samples — training format now
     matches pipeline.py inference format exactly (system + user + assistant).
  5. RAG-aware training samples: golden QA formatted with retrieved
     context in the user turn, teaching the model to ground answers
     in source chunks rather than parametric memory.

Sources in priority order:
  1. golden_qa.json        — curated QA (5x weight, highest signal)
  2. reddit_questions.json — real-user questions (1x weight)
  3. all_chunks.json       — synthetic QA from corpus (1x weight)
  4. Instruction samples   — general capability preservation (~50 samples)
"""

import json
import random
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
GOLDEN_QA      = PROJECT_ROOT / "data" / "evaluation" / "golden_qa.json"
REDDIT_QA      = PROJECT_ROOT / "data" / "evaluation" / "reddit_questions.json"
CHUNKS_PATH    = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"
OUT_DIR        = PROJECT_ROOT / "Fine-Tuneing"
TRAIN_OUT      = OUT_DIR / "train.txt"
VAL_OUT        = OUT_DIR / "val.txt"

# ── Config ────────────────────────────────────────────────────────────────────
GOLDEN_WEIGHT    = 5
REDDIT_WEIGHT    = 1
CHUNK_WEIGHT     = 1

VAL_SPLIT        = 0.10
RANDOM_SEED      = 42

MIN_CHUNK_LEN    = 50
MIN_WORD_COUNT   = 8
MAX_CHUNK_LEN    = 3200
MAX_ANSWER_CHARS = 800

# System prompt included in training samples so format matches pipeline.py exactly.
# Training format: system + user + assistant
# Inference format: system (from pipeline.py) + user + assistant
# Mismatch between the two was the root cause of relevancy degradation.
# Fix 1: SYSTEM_CONTEXT matches the preamble of prompts.py SYSTEM_PROMPT exactly.
# Inference format (prompts.py):
#   "You are a legal information assistant for Massachusetts tenant law (Boston area).
#    RULES: 1. ONLY answer from provided source documents...
#    CONTEXT: {context}
#    QUESTION: {question}"
# Training must use the same role/rules preamble so the model learns the correct
# behavior under the exact prompt it will see at inference time.
# CRITICAL: Must match pipeline.py SYSTEM_PROMPT exactly (minus {context}/{question}
# placeholders). Training/inference format mismatch causes model incoherence.
SYSTEM_CONTEXT = """You are a retrieval-grounded legal information assistant for Massachusetts tenant law (Boston area).

RULES:
1. Use ONLY the retrieved context below to answer. Do not use outside knowledge.
2. NEVER provide legal ADVICE -- only legal INFORMATION. Recommend consulting an attorney for specific situations.
3. ALWAYS cite sources with [Source: <title> (<url>)]. Cite specific statutes (e.g., MGL c.186, s.15B) when they appear in the context.
4. If the retrieved context is insufficient or conflicting, say so clearly and suggest legal aid resources such as MassLegalHelp.org or Greater Boston Legal Services.
5. If the question is outside Massachusetts tenant law, say so."""


# ── Answer truncation ─────────────────────────────────────────────────────────
def truncate_answer(text: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    """Truncate at last paragraph or sentence boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    last_para = window.rfind("\n\n")
    if last_para > max_chars // 2:
        return text[:last_para].strip()
    last_sent = max(
        window.rfind(". "), window.rfind(".\n"),
        window.rfind("! "), window.rfind("? "),
    )
    if last_sent > max_chars // 2:
        return text[:last_sent + 1].strip()
    return window.strip()


# ── Sample formatting ───────────────────────────────────────────────────────
def format_sample(question: str, answer: str) -> str:
    """Format as Qwen3 ChatML WITH system prompt.

    System prompt is included to match inference format — pipeline.py injects
    it at inference time, so training must match or the model sees a format
    it was never trained on and becomes incoherent.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_CONTEXT}<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer.strip()}<|im_end|>"
    )


def format_rag_sample(question: str, context: str, answer: str) -> str:
    """RAG-aware format matching pipeline.py inference format exactly.

    pipeline.py: system_msg = SYSTEM_PROMPT.format(context=context, question=question)
    which inserts context and question into the SYSTEM role, then sends the
    question again as the USER role. We replicate that token structure here.
    """
    # Build the system message exactly as pipeline.py does:
    # SYSTEM_PROMPT.format(context=context, question=question)
    # SYSTEM_CONTEXT has a Respond in this format: section at the end
    # We insert RETRIEVED CONTEXT + QUESTION before that section
    # Insert RETRIEVED CONTEXT + QUESTION into system message,
    # matching pipeline.py's SYSTEM_PROMPT.format(context=context, question=question)
    system_msg = (
        SYSTEM_CONTEXT
        + f"\n\nRETRIEVED CONTEXT:\n{context.strip()}\n\nQUESTION: {question.strip()}"
    )
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer.strip()}<|im_end|>"
    )

def format_instruction_sample(question: str, answer: str) -> str:
    """Format instruction-following preservation samples WITHOUT domain system prompt.

    These samples are general capability Q&A (arithmetic, grammar, definitions).
    Using the MA tenant law system prompt here would be contradictory — the model
    would learn to answer off-topic questions under the legal prompt, or learn to
    ignore the system prompt entirely. No system prompt = clean general instruction
    following that doesn't interfere with domain training.
    """
    return (
        f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer.strip()}<|im_end|>"
    )


# ── Redesign 1: instruction-following preservation samples ────────────────────
def load_instruction_samples() -> list[dict]:
    """~50 general instruction-following samples.

    Redesign 1: prevents LoRA from eroding base model capabilities.
    These are NOT domain-specific — they exercise the model's general
    ability to follow instructions, summarize, format, and reason.
    Ratio target: ~10% of total dataset.
    """
    samples = [
        # Following instructions
        ("Please summarize the following in one sentence: The landlord must return the security deposit within 30 days of the tenant vacating the premises, along with any accrued interest.",
         "The landlord must return the security deposit plus interest within 30 days of the tenant moving out."),
        ("Answer with YES or NO only: Is Massachusetts a state in the United States?", "YES"),
        ("Answer with YES or NO only: Is a security deposit the same as first month's rent?", "NO"),
        ("List three things a tenant should do before moving out of a rental.", "1. Document the condition of the apartment with photos or video.\n2. Provide written notice to the landlord per the lease terms.\n3. Return all keys and get written confirmation of their return."),
        ("What is 30 multiplied by 3?", "90"),
        ("Convert this to a bullet list: Tenants have the right to a habitable unit, the right to privacy, and the right to proper notice before entry.", "- Right to a habitable unit\n- Right to privacy\n- Right to proper notice before entry"),
        ("Rewrite this sentence more clearly: The landlord by law must give the tenant before entering notice.", "By law, the landlord must give the tenant notice before entering."),
        ("What does MGL stand for in Massachusetts law?", "MGL stands for Massachusetts General Laws, which is the codified collection of permanent laws for the Commonwealth of Massachusetts."),
        ("Is this statement true or false: A landlord can charge any amount for a security deposit in Massachusetts.", "False. Massachusetts law limits security deposits to no more than one month's rent."),
        ("Explain the difference between a lease and a tenancy at will.", "A lease is a fixed-term agreement (e.g., one year) that binds both parties for a set period. A tenancy at will has no fixed end date and can be terminated by either party with proper notice."),
        # Declining gracefully out of scope
        ("What is the capital of France?", "Paris is the capital of France. Note that this is outside my area of focus — I specialize in Massachusetts tenant law."),
        ("Can you write me a poem about spring?", "I specialize in Massachusetts tenant law questions. For creative writing requests, a general-purpose assistant would serve you better."),
        ("What is the best recipe for chocolate chip cookies?", "I focus on Massachusetts tenant law questions and am not the right resource for recipes. A general-purpose assistant or cooking website would help more."),
        # Citing uncertainty
        ("What will the Massachusetts eviction moratorium policy be in 2030?", "I cannot predict future policy changes. For current Massachusetts eviction laws, please consult mass.gov or a licensed Massachusetts attorney."),
        ("Can you tell me the outcome of my specific court case?", "I cannot predict or advise on specific legal case outcomes. For your situation, please consult a licensed Massachusetts attorney or contact Greater Boston Legal Services."),
        # Formatting
        ("Format this as a numbered list: first you must provide written notice, then you must wait the required notice period, then you may proceed with eviction.", "1. Provide written notice to the tenant.\n2. Wait the required notice period.\n3. Proceed with the eviction process."),
        ("What does 'habitable' mean in the context of housing?", "In housing law, 'habitable' means the unit meets minimum standards for human occupation — including functional heat, water, electricity, structural safety, and freedom from pests or hazardous conditions."),
        ("Translate this legal term into plain English: 'tenancy at sufferance'.", "A tenancy at sufferance occurs when a tenant stays in a rental after their lease has expired without the landlord's permission. The landlord can choose to evict them or accept rent to create a new tenancy."),
        ("Give me a one-sentence definition of 'quiet enjoyment'.", "Quiet enjoyment is a tenant's right to use and enjoy their rental property without interference from the landlord."),
        ("What is the difference between civil and criminal law?", "Civil law governs disputes between individuals or organizations (such as landlord-tenant disagreements), while criminal law involves offenses against the state and can result in fines or imprisonment."),
        # More instruction types
        ("Respond in exactly two sentences: What is rent withholding?", "Rent withholding is a legal remedy where a tenant stops paying rent due to serious habitability problems the landlord has failed to fix. In Massachusetts, tenants must follow specific legal procedures before withholding rent."),
        ("What are the first three letters of the alphabet?", "A, B, C."),
        ("If a tenant pays $1,500 per month in rent, what is the maximum security deposit a Massachusetts landlord can charge?", "The maximum security deposit is $1,500 — equal to one month's rent, which is the legal limit under Massachusetts law."),
        ("Summarize in plain English: MGL c.186 s.15B requires landlords to hold security deposits in a separate, interest-bearing account.", "Massachusetts law requires landlords to keep security deposits in their own bank account that earns interest, separate from the landlord's own money."),
        ("What does 'pro rata' mean?", "Pro rata means proportional — for example, if a tenant moves in mid-month, they pay a pro rata share of rent for just the days they occupied the unit."),
        # Edge cases
        ("I need legal advice about my specific situation.", "I can provide general legal information about Massachusetts tenant law, but I cannot give legal advice for your specific situation. For personalized guidance, please consult a licensed Massachusetts attorney or contact a legal aid organization like Greater Boston Legal Services."),
        ("Are you a lawyer?", "No, I am not a lawyer. I provide general legal information about Massachusetts tenant law. For legal advice specific to your situation, please consult a licensed Massachusetts attorney."),
        ("What is your name?", "I am a Massachusetts tenant law information assistant. I can help answer general questions about renter rights, landlord obligations, evictions, security deposits, and related topics under Massachusetts law."),
        ("Can you help me draft a lease agreement?", "I can explain what Massachusetts law requires in a lease, but drafting a specific legal document is outside my scope. I recommend consulting a Massachusetts attorney for document drafting."),
        ("How do I contact my landlord?", "You should contact your landlord using the method specified in your lease. For formal notices — such as repair requests or notice to quit — Massachusetts law recommends written notice sent by certified mail to create a paper trail."),
        ("Translate 'notice to quit' into plain English.", "A notice to quit is a formal written document a landlord gives a tenant demanding they leave the rental property. It is the first legal step in the eviction process in Massachusetts."),
        ("What is 'triple damages' in tenant law?", "Triple damages means a court can order a landlord to pay three times the amount of harm caused to the tenant — for example, three times the wrongfully withheld security deposit — as a penalty for violating the law."),
        ("Give a one-word answer: Is heat required in Massachusetts rental units?", "Yes."),
        ("What does 'waive' mean in a legal context?", "To waive a right means to voluntarily give it up. For example, a landlord who accepts rent after serving a notice to quit may be considered to have waived their right to evict based on that notice."),
        ("How many days are in 30 days?", "30 days."),
        ("What agency enforces housing codes in Boston?", "The Inspectional Services Department (ISD) enforces housing codes in Boston. Tenants can file complaints at 617-635-5300 or online at boston.gov/isd."),
        ("Define 'implied warranty of habitability'.", "The implied warranty of habitability is a legal guarantee that a landlord must maintain a rental unit in a livable condition meeting basic health and safety standards, even if the lease does not explicitly say so."),
        ("What is 'constructive eviction'?", "Constructive eviction occurs when a landlord's actions or inactions make the rental unit so uninhabitable that the tenant is effectively forced to leave, even without a formal eviction notice."),
        ("What does 'at-will' mean in tenancy?", "An at-will tenancy has no fixed end date. Either the landlord or tenant can end it by giving proper notice — typically 30 days in Massachusetts for a month-to-month arrangement."),
        ("Can a landlord enter my apartment whenever they want?", "No. In Massachusetts, landlords must give reasonable notice before entering a tenant's unit except in emergencies. Repeated uninvited entry may violate the tenant's right to quiet enjoyment."),
        ("What is a 'security deposit'?", "A security deposit is money a tenant pays to a landlord before moving in, held as financial protection against unpaid rent or damages beyond normal wear and tear. Massachusetts limits it to one month's rent."),
        ("What does 'notice period' mean?", "A notice period is the required amount of time a landlord or tenant must give before terminating a tenancy or taking certain actions. For example, a month-to-month tenant in Massachusetts typically requires 30 days notice."),
        ("Explain 'right of entry' in one sentence.", "The right of entry is the landlord's legal ability to access the rental unit, which in Massachusetts requires reasonable advance notice to the tenant except in emergencies."),
        ("What is mediation in housing disputes?", "Mediation is a process where a neutral third party helps a landlord and tenant resolve a dispute without going to court. Boston and many Massachusetts cities offer free or low-cost housing mediation services."),
        ("What is a housing court?", "A housing court is a specialized court that handles landlord-tenant disputes, evictions, and housing code violations. Massachusetts has seven housing courts that cover the entire state."),
        ("What does 'plaintiff' mean?", "The plaintiff is the party who initiates a lawsuit. In an eviction case, the landlord is typically the plaintiff."),
        ("What does 'defendant' mean?", "The defendant is the party being sued. In an eviction case, the tenant is typically the defendant."),
        ("What is a 'demand letter'?", "A demand letter is a formal written request — typically sent before filing a lawsuit — asking the other party to take a specific action, such as returning a security deposit or making repairs."),
        ("What is 'small claims court'?", "Small claims court is a simplified court process for resolving disputes involving smaller amounts of money. In Massachusetts, tenants can use small claims court to recover wrongfully withheld security deposits."),
        ("Can a verbal lease be valid in Massachusetts?", "Yes, a verbal lease can be legally valid in Massachusetts, but written leases are strongly recommended because they provide clear documentation of the agreed terms."),
        ("What is 'normal wear and tear'?", "Normal wear and tear refers to the gradual deterioration of a rental unit from ordinary use over time — such as minor scuffs on walls or worn carpet. Landlords cannot deduct from a security deposit for normal wear and tear."),
    ]

    pairs = []
    for q, a in samples:
        pairs.append({"question": q, "answer": a, "source": "instruction"})

    print(f"  Loaded {len(pairs)} instruction-following preservation samples")
    return pairs


def load_refusal_samples() -> list[dict]:
    """~15 out-of-scope refusal samples WITH the domain system prompt.

    Fix 6: teaches the model how to respond when a question is out of scope
    while operating under the MA tenant law system prompt. Without these,
    the model has no training signal for refusals and may hallucinate or
    answer off-topic questions as if they were legal ones.
    """
    samples = [
        # Out of scope — wrong state
        ("My landlord in California didn't return my security deposit. What can I do?",
         "I specialize in Massachusetts tenant law and cannot advise on California landlord-tenant issues. For California-specific help, contact your local tenant rights organization or the California Department of Consumer Affairs."),

        # Out of scope — commercial lease
        ("I run a small business and my commercial landlord is raising my rent. What are my rights?",
         "I only cover residential tenant law in Massachusetts. Commercial leases are governed by contract law and are outside this system's scope. For commercial lease disputes, please consult a business attorney."),

        # Out of scope — federal law
        ("What are my rights under the federal Fair Housing Act?",
         "My sources cover Massachusetts state law. For federal Fair Housing Act information, visit hud.gov or contact a local fair housing organization. I can answer questions about Massachusetts fair housing law under MGL c.151B if that would help."),

        # Insufficient context
        ("Can my landlord evict me?",
         "In Massachusetts, landlords cannot evict tenants without following a specific legal process: written notice, a waiting period, and a court-ordered judgment. The grounds and timeline vary by situation — nonpayment, lease violation, and end of tenancy each have different procedures. Could you share more details about your circumstances?"),

        # Asks for legal advice
        ("Should I sue my landlord?",
         "I can provide legal information about tenant rights in Massachusetts, but whether you should sue is a legal decision I cannot make for you. I strongly recommend consulting a Massachusetts attorney or contacting Greater Boston Legal Services (617-603-1700) for advice specific to your situation."),

        # Completely unrelated
        ("What is the best way to invest in real estate?",
         "I specialize in Massachusetts residential tenant law — questions about security deposits, repairs, evictions, and similar topics. Real estate investment advice is outside my scope. A financial advisor would be better suited to help."),

        # Asks about landlord rights (not tenant)
        ("I am a landlord. How do I evict a tenant who hasn't paid rent?",
         "In Massachusetts, to evict for nonpayment you must: (1) serve a 14-day notice to quit, (2) wait for the period to expire, (3) file a summary process complaint in Housing Court, and (4) attend a hearing. Self-help evictions such as changing locks or removing belongings are illegal. Both landlords and tenants should consult an attorney for their specific situation."),

        # No context retrieved scenario
        ("What is the maximum late fee a landlord can charge in Massachusetts?",
         "The retrieved sources don't cover this specific question. Generally, Massachusetts law limits when late fees can be charged, but I cannot give you an accurate figure without the relevant source. I recommend checking MassLegalHelp.org or consulting a tenant attorney for the current limit."),

        # Wants prediction
        ("Will my landlord win the eviction case?",
         "I cannot predict the outcome of your case — that would be legal advice, which I'm not able to provide. What I can tell you is what Massachusetts law says about the eviction process and tenant defenses. For case-specific guidance, please contact Greater Boston Legal Services (617-603-1700) or a Massachusetts attorney."),

        # Asks about another country
        ("I live in Toronto and my landlord won't fix my heat. What are my rights?",
         "I specialize in Massachusetts (USA) tenant law and cannot advise on Canadian housing law. For Ontario tenant rights, contact the Landlord and Tenant Board of Ontario at tribunalsontario.ca/ltb."),
    ]

    pairs = []
    for q, a in samples:
        pairs.append({"question": q, "answer": a, "source": "refusal"})

    print(f"  Loaded {len(pairs)} refusal/out-of-scope samples")
    return pairs


# ── Redesign 5: RAG-aware samples from golden QA + source chunks ──────────────
def load_rag_samples() -> list[dict]:
    """Golden QA formatted with source chunk context in the user turn.

    Redesign 5: teaches the model to read retrieved chunks and ground its
    answer in them. Addresses the rerank degradation where the model was
    ignoring context. Uses golden_qa.json source_chunks field.
    """
    if not GOLDEN_QA.exists():
        return []

    with open(GOLDEN_QA, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not CHUNKS_PATH.exists():
        return []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunk_by_id = {c["chunk_id"]: c for c in all_chunks}

    pairs = []
    for item in data:
        question = item.get("question", "").strip()
        answer   = item.get("expected_answer", "").strip()
        if not question or not answer:
            continue

        source_chunks = item.get("source_chunks", [])
        if not source_chunks:
            continue

        # Build context using same format as prompts.py format_context():
        #   [Source N: title (url)]\ncontent  joined by \n\n---\n\n
        context_parts = []
        for i, sc in enumerate(source_chunks, 1):
            chunk_id = sc.get("chunk_id", "")
            chunk    = chunk_by_id.get(chunk_id)
            if chunk:
                context_parts.append(
                    f"[Source {i}: {chunk['title']} ({chunk['source_url']})]\n"
                    f"{chunk['content']}"
                )

        if not context_parts:
            continue

        context = "\n\n---\n\n".join(context_parts)
        pairs.append({
            "question": question,
            "answer":   answer,
            "context":  context,
            "source":   "rag",
        })

    print(f"  Loaded {len(pairs)} RAG-aware samples (golden QA + source chunks)")
    return pairs


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_golden_qa() -> list[dict]:
    if not GOLDEN_QA.exists():
        print(f"  [WARN] {GOLDEN_QA} not found, skipping golden QA")
        return []

    with open(GOLDEN_QA, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        question = item.get("question", "").strip()
        if not question:
            continue
        answer = item.get("expected_answer", "").strip()
        if not answer and "key_facts" in item:
            facts = item["key_facts"]
            if isinstance(facts, list):
                answer = " ".join(str(f) for f in facts).strip()
        if answer:
            pairs.append({"question": question, "answer": answer, "source": "golden"})

    print(f"  Loaded {len(pairs)} golden QA pairs")
    return pairs


def load_reddit_qa() -> list[dict]:
    if not REDDIT_QA.exists():
        print(f"  [WARN] {REDDIT_QA} not found, skipping Reddit QA")
        return []

    with open(REDDIT_QA, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        question = item.get("question", "").strip()
        if not question:
            continue
        answer = (
            item.get("expected_answer")
            or item.get("enriched_answer")
            or item.get("ideal_answer")
            or item.get("answer")
            or ""
        ).strip()
        # Do NOT prepend context to question — it creates an unusually long
        # user turn that doesn't match real inference-time questions.
        # The question alone is more representative of actual user input.
        if answer:
            pairs.append({"question": question, "answer": answer, "source": "reddit"})

    print(f"  Loaded {len(pairs)} Reddit QA pairs")
    return pairs


def chunk_to_qa(chunk: dict) -> dict | None:
    content = chunk.get("content", "").strip()
    if len(content) < MIN_CHUNK_LEN or len(content) > MAX_CHUNK_LEN:
        return None
    if len(content.split()) < MIN_WORD_COUNT:
        return None

    title        = chunk.get("title", "").strip()
    source_name  = chunk.get("source_name", "").strip()
    content_type = chunk.get("content_type", "").strip()

    if content_type == "faq":
        lines      = content.splitlines()
        first_line = lines[0].strip() if lines else ""
        question   = first_line if first_line.endswith("?") else f"What does {source_name} say about {title.lower()}?"
    elif content_type in ("statute", "regulation"):
        statute_match = re.search(r"(MGL\s+c\.\s*\d+[A-Z]?|CMR\s+\d+)", content)
        question = (
            f"What does {statute_match.group(1)} say about tenant rights in Massachusetts?"
            if statute_match else f"What are the legal requirements under {title}?"
        )
    elif content_type == "guide":
        question = f"What guidance does {source_name} provide about {title.lower()}?"
    else:
        question = f"What does Massachusetts law say about {title.lower()}?"

    return {"question": question, "answer": truncate_answer(content), "source": "chunk"}


def load_chunk_qa() -> list[dict]:
    if not CHUNKS_PATH.exists():
        print(f"  [WARN] {CHUNKS_PATH} not found, skipping chunk QA")
        return []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    pairs, skipped = [], 0
    for chunk in chunks:
        qa = chunk_to_qa(chunk)
        if qa:
            pairs.append(qa)
        else:
            skipped += 1

    before_dedup = len(pairs)
    seen = {}
    for qa in pairs:
        q_norm = re.sub(r"\s+", " ", qa["question"].strip().lower())
        if q_norm not in seen or len(qa["answer"]) > len(seen[q_norm]["answer"]):
            seen[q_norm] = qa
    pairs = list(seen.values())

    print(f"  Converted {before_dedup} chunks to QA pairs ({skipped} skipped)")
    print(f"  Deduplicated: {before_dedup} -> {len(pairs)} (-{before_dedup - len(pairs)})")
    return pairs


# ── Assembly ──────────────────────────────────────────────────────────────────
def build_dataset() -> list[str]:
    print("\nLoading training data sources...")
    golden      = load_golden_qa()
    reddit      = load_reddit_qa()
    chunks      = load_chunk_qa()
    instruction = load_instruction_samples()   # Fix 1: capability preservation
    rag         = load_rag_samples()           # Fix 5: RAG grounding
    refusal     = load_refusal_samples()       # Fix 6: out-of-scope handling

    # Weight domain samples
    weighted = (
        golden      * GOLDEN_WEIGHT +
        reddit      * REDDIT_WEIGHT +
        chunks      * CHUNK_WEIGHT  +
        instruction * 1  +   # 1x — general capability preservation
        rag         * 3  +   # 3x — RAG grounding, critical for rerank configs
        refusal     * 2  +   # 2x — out-of-scope handling under domain system prompt
        []
    )

    print(f"\nDataset composition:")
    print(f"  Golden QA:      {len(golden)} x {GOLDEN_WEIGHT} = {len(golden) * GOLDEN_WEIGHT} samples")
    print(f"  Reddit QA:      {len(reddit)} x {REDDIT_WEIGHT} = {len(reddit) * REDDIT_WEIGHT} samples")
    print(f"  Chunk QA:       {len(chunks)} x {CHUNK_WEIGHT} = {len(chunks) * CHUNK_WEIGHT} samples")
    print(f"  Instruction:    {len(instruction)} x 1 = {len(instruction)} samples (capability preservation)")
    print(f"  RAG-aware:      {len(rag)} x 3 = {len(rag) * 3} samples (grounding)")
    print(f"  Refusal:        {len(refusal)} x 2 = {len(refusal) * 2} samples (out-of-scope handling)")
    print(f"  Total:          {len(weighted)} samples")

    random.seed(RANDOM_SEED)
    random.shuffle(weighted)

    # Redesign 3: format WITHOUT system prompt
    # Redesign 5: RAG samples use format_rag_sample
    formatted = []
    for p in weighted:
        if p["source"] == "rag":
            # RAG samples: system prompt + context in user turn
            formatted.append(format_rag_sample(p["question"], p["context"], p["answer"]))
        elif p["source"] == "instruction":
            # Instruction samples: no system prompt — general capability, not domain
            formatted.append(format_instruction_sample(p["question"], p["answer"]))
        elif p["source"] == "refusal":
            # Refusal samples: full system prompt — model learns to decline under domain prompt
            formatted.append(format_sample(p["question"], p["answer"]))
        else:
            # Domain samples (golden, reddit, chunk): full system prompt
            formatted.append(format_sample(p["question"], p["answer"]))

    return formatted


def write_splits(samples: list[str]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_val         = max(1, int(len(samples) * VAL_SPLIT))
    n_train       = len(samples) - n_val
    train_samples = samples[:n_train]
    val_samples   = samples[n_train:]

    # Use a sentinel separator that cannot appear inside samples.
    # \n\n breaks when RAG samples contain multi-paragraph retrieved context.
    SENTINEL = "\n<<<SAMPLE_BOUNDARY>>>\n"
    TRAIN_OUT.write_text(SENTINEL.join(train_samples), encoding="utf-8")
    VAL_OUT.write_text(  SENTINEL.join(val_samples),   encoding="utf-8")

    print(f"\nWrote training data:")
    print(f"  Train: {len(train_samples)} samples -> {TRAIN_OUT}")
    print(f"  Val:   {len(val_samples)} samples   -> {VAL_OUT}")
    print(f"  Train file size: {TRAIN_OUT.stat().st_size / 1024:.1f} KB")
    print(f"  Val file size:   {VAL_OUT.stat().st_size / 1024:.1f} KB")


def preview(samples: list[str], n: int = 3):
    print(f"\n{'=' * 70}")
    print(f"Sample preview ({n} samples):")
    print(f"{'=' * 70}")
    for i, s in enumerate(samples[:n], 1):
        print(f"\n--- Sample {i} ---")
        print(s[:600] + "..." if len(s) > 600 else s)


if __name__ == "__main__":
    import sys

    print("MA Tenant Law -- Finetune Data Preparation (Qwen3 ChatML format)")
    print("Redesigns: 1 (instruction preservation), 3 (no system prompt), 5 (RAG-aware)")
    print("=" * 60)

    samples = build_dataset()

    if "--preview" in sys.argv or "-p" in sys.argv:
        preview(samples)

    write_splits(samples)

    print("\nDone. Next step:")
    print("  python Fine-Tuneing/FineTune.py")
