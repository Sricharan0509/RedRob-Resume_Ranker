"""
config.py — single source of truth for the **RAG candidate ranker**.

The ranker implements a "Hybrid Knowledge-Graph & Multi-Stage LLM Reranking"
blueprint, made guardrail-compliant (CPU-only, no network / no hosted LLM at
rank time) by using a *local* vector store + *local* cross-encoder:

    Phase 1  Parsing & contextual feature extraction        (data_processing/)
    Phase 2  Dual-engine retrieval: dense vector store + BM25 (retrieval/)
    Phase 3  Recruiter-brain scoring (structured rubric + behavioral) (reranking/recruiter_brain)
    Phase 4  Cross-encoder reranking of the shortlist          (reranking/cross_encoder)

Scoring blend
-------------
    Sim_hybrid      = W_DENSE*S_dense + W_LEX*S_lex          # Phase-2 retrieval score
    recruiter_fit   = W_STRUCT*S_struct + W_SIM*Sim_hybrid   # Phase-3 composite
    shortlist (top SHORTLIST_N by recruiter_fit) is reranked:
        fit         = (1-CROSS_BLEND)*recruiter_fit + CROSS_BLEND*S_cross
    base            = role_gate * fit
    final           = base * behavioral_mult * honeypot_kill

Everything tunable lives here. No magic numbers scattered in logic.
"""

from datetime import date

# Reference "today" for all recency/tenure date math.
TODAY = date(2026, 7, 2)
SEED = 42  # seed everything → byte-identical output

# ── Phase-2 retrieval fusion (dense + lexical) ───────────────────────────────
W_DENSE = 0.60   # dense semantic cosine from the vector store
W_LEX = 0.40     # BM25 lexical over hard-constraint terms
assert abs(W_DENSE + W_LEX - 1.0) < 1e-9

# ── Phase-3 recruiter-brain composite (Score_Final backbone) ─────────────────
#   recruiter_fit = W_STRUCT*S_struct + W_SIM*Sim_hybrid
W_STRUCT = 0.62  # structured "recruiter brain" rubric — anti-stuffer backbone
W_SIM = 0.38     # semantic+lexical retrieval similarity
assert abs(W_STRUCT + W_SIM - 1.0) < 1e-9

# ── Phase-4 cross-encoder rerank ─────────────────────────────────────────────
SHORTLIST_N = 300          # candidates re-scored by the local cross-encoder
CROSS_BLEND = 0.50         # weight of S_cross within the shortlist's fit

# ── Structured "recruiter brain" rubric weights ─────────────────────────────
STRUCT_WEIGHTS = dict(
    role_fit=0.28,          # title/trajectory seniority + relevance
    career_evidence=0.24,   # what they actually built (career descriptions)
    skill_trust=0.18,       # trust-weighted JD-relevant skills
    experience_fit=0.12,    # soft curve, peak 6–8 yrs
    eval_framework=0.08,    # NDCG/MRR/MAP/A-B literacy
    product_company=0.10,   # product co. / meaningful scale
)

# Structured penalties (subtracted from S_struct, each capped in code)
PENALTIES = dict(
    cv_speech_robotics_primary=0.25,
    academic_only_no_prod=0.30,
    langchain_recent_only=0.20,
    title_chaser=0.15,
    consulting_only_career=0.20,
    # JD "What we mean by 5-9 years": "a senior engineer who hasn't written
    # production code in the last 18 months because you've moved into
    # 'architecture' or 'tech lead' roles" — same "probably not move forward"
    # strength as langchain_recent_only, so weighted the same.
    architecture_tech_lead_stale=0.20,
)

# ── role_gate tiers — multiplicative, uses title ∪ career evidence (NOT skills)
ROLE_GATE_CORE = 1.00       # core AI/ML/DS/IR/Search/RecSys/NLP title
ROLE_GATE_ADJ_EVID = 0.95   # adjacent title BUT career shows retrieval/ranking built
ROLE_GATE_ADJ_TECH = 0.70   # adjacent tech (SWE/Backend/Data/DevOps) w/ some evidence
ROLE_GATE_GENERIC = 0.30    # generic tech, weak/no evidence
ROLE_GATE_NONTECH = 0.05    # non-tech / clearly irrelevant

# ── behavioral_mult (availability) ───────────────────────────────────────────
BEHAV_BASE = 0.85
BEHAV_MIN = 0.55
BEHAV_MAX = 1.15

# ── honeypot kill factor ─────────────────────────────────────────────────────
HONEYPOT_KILL = 0.001

# ── Models (all local, CPU, offline) ─────────────────────────────────────────
DENSE_MODEL = "BAAI/bge-small-en-v1.5"           # 384-d bi-encoder
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DENSE_DIM = 384
FALLBACK_DIM = 512   # dim of the pure-numpy hashing encoder used if ST is absent

# Dense-channel encoder selection (Phase 2 bi-encoder in build_index.py / the
# in-budget fallback in rank.py):
#   "hash" - ALWAYS use the deterministic pure-numpy hashing encoder. Builds the
#            100K index in ~5 min on CPU, no model download, fully reproducible.
#   "auto" - use real BGE-small (sentence-transformers) when importable, else
#            fall back to hash. Real BGE encodes 100K in ~2.75h on this CPU.
# Set to "hash" deliberately: the ~2.75h real-BGE build is not worth the wall
# time here, and S_struct (0.62 weight) + BM25 + the cross-encoder carry ranking
# quality. This does NOT affect the Phase-4 cross-encoder (a separate, fast
# model that only scores the 300-shortlist at rank time).
DENSE_ENCODER_MODE = "hash"      # {"hash", "auto"}

# ── CPU safety knobs (guards a confirmed OOM: batch_size=256 x the model's
#    default max_seq_length=512 is a single ~3GB attention-matrix allocation
#    per layer on CPU) ──────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 256         # cap on tokens/sequence for ST bi-/cross-encoders
ENCODE_BATCH_SIZE = 128      # bi-encoder batch size (build_index.py, fallback path)
CROSS_BATCH_SIZE = 32        # cross-encoder batch size (rank-time, 300-shortlist)
USE_ALL_CPU_THREADS = True   # torch defaults to 4 threads; use all cores instead

# BGE wants a query-instruction prefix on the *query* side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Vector store ─────────────────────────────────────────────────────────────
#   "chroma" if chromadb is installed at build time; else a numpy flat index.
VECTOR_STORE_BACKEND = "auto"   # {"auto","chroma","numpy"}
CHROMA_COLLECTION = "redrob_candidates"

# ── Artifact paths (relative to the package root) ────────────────────────────
ARTIFACT_DIR = "artifacts"
DENSE_NPY = "dense.f16.npy"     # (N, dim) float16 profile embeddings
IDS_NPY = "ids.npy"            # (N,) candidate_id order, aligned to dense.npy
# The BM25 query (JD_BM25_TERMS below) is fixed at build time, so we persist
# the resulting per-candidate score vector rather than the full tokenized
# postings index (~130MB pickled for 100K docs -- over GitHub's 100MB
# per-file push limit). Aligned to ids.npy, same as dense.f16.npy.
BM25_SCORES_NPY = "bm25_scores.npy"  # (N,) float32 BM25 score vs JD_BM25_TERMS
JD_DENSE_NPY = "jd_dense.npy"  # (dim,) mean-pooled JD query vector
META_JSON = "build_meta.json"  # which encoder/backend produced the artifacts

# ── profile_doc weighting — career descriptions up-weighted ──────────────────
DOC_W_HEADLINE_SUMMARY = 1
DOC_W_CAREER_DESC = 2
DOC_W_SKILLS = 1
DOC_MAX_CHARS = 4000

# ─────────────────────────────────────────────────────────────────────────────
# JD-derived query text
# ─────────────────────────────────────────────────────────────────────────────

# Dense query: 3 positive "ideal candidate" passages, mean-pooled.
# Deliberately NO anti-patterns embedded here.
JD_DENSE_PASSAGES = [
    "Senior AI engineer who has shipped production embedding-based retrieval, "
    "ranking, and recommendation systems to real users at meaningful scale.",
    "Built dense retrieval and hybrid search on vector databases such as FAISS, "
    "Pinecone, Weaviate, Qdrant or Elasticsearch; handled embedding drift, index "
    "refresh, and retrieval-quality regression in production.",
    "Designs rigorous evaluation for ranking systems with NDCG, MRR, MAP, "
    "offline-to-online correlation and A/B testing; strong Python; some LLM "
    "fine-tuning with LoRA or QLoRA.",
]

# Cross-encoder query: one compact JD passage paired against each profile_doc.
JD_CROSS_TEXT = (
    "Senior AI Engineer for a product company. Needs production experience with "
    "embeddings-based retrieval (sentence-transformers, BGE, E5), vector databases "
    "and hybrid search (FAISS, Pinecone, Weaviate, Qdrant, Elasticsearch), strong "
    "Python, and evaluation frameworks for ranking (NDCG, MRR, MAP, A/B testing). "
    "Has shipped an end-to-end search, ranking, or recommendation system. Not a "
    "keyword-stuffer, not pure research, not consulting-only, not primarily computer "
    "vision or speech."
)

# BM25 query: curated JD hard-constraint terms.
JD_BM25_TERMS = [
    "embeddings", "retrieval", "ranking", "vector", "search", "faiss",
    "pinecone", "weaviate", "qdrant", "elasticsearch", "ndcg", "mrr", "map",
    "recommendation", "information", "hybrid", "reranking", "rerank",
    "production", "deployed", "sentence", "transformers", "semantic",
    "learning", "rank", "bm25", "recsys", "embedding",
]

# ─────────────────────────────────────────────────────────────────────────────
# Title taxonomy
# ─────────────────────────────────────────────────────────────────────────────
AI_TITLES = frozenset({
    "ml engineer", "machine learning engineer", "ai engineer",
    "ai research engineer", "nlp engineer", "search engineer",
    "ranking engineer", "recommendation engineer", "applied scientist",
    "research engineer", "applied ml", "applied ai",
    "deep learning engineer", "llm engineer", "generative ai engineer",
    "information retrieval", "recsys",
    "recommendation systems engineer",  # "recommendation engineer" doesn't
                                         # match "...systems engineer"
    "ai specialist",
})
DATA_SCIENTIST_TITLES = frozenset({
    "data scientist", "senior data scientist", "lead data scientist",
    "principal data scientist",
})
ADJACENT_TITLES = frozenset({
    "analytics engineer", "data engineer", "senior data engineer",
    "backend engineer", "software engineer", "senior software engineer",
    "staff engineer", "principal engineer", "platform engineer",
    "cloud engineer", "full stack developer", "java developer",
    ".net developer", "devops engineer", "mobile developer",
    "frontend engineer", "qa engineer", "data analyst",
    "senior data analyst",
})
NON_TECHNICAL_TITLES = frozenset({
    "marketing manager", "hr manager", "human resources",
    "accountant", "content writer", "sales executive", "graphic designer",
    "customer support", "business analyst", "civil engineer",
    "mechanical engineer", "operations manager", "project manager",
    "product designer", "ux designer", "recruiter", "talent acquisition",
})

# JD's own words for the "moved off hands-on coding" disqualifier: "you've
# moved into 'architecture' or 'tech lead' roles". Matched against the
# CURRENT job's title only (is_current=True), not the title taxonomy above,
# since "principal/staff engineer" (ADJACENT_TITLES) are still hands-on ICs.
ARCHITECTURE_TECH_LEAD_KW = ("architect", "tech lead", "technical lead")

# ── Skill taxonomies ─────────────────────────────────────────────────────────
CRITICAL_SKILLS = frozenset({
    "sentence-transformers", "sentence transformers", "embeddings",
    "semantic search", "dense retrieval", "bi-encoder", "cross-encoder",
    "openai embeddings", "bge", "e5", "text embeddings", "embedding model",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "vector database", "vector search", "vector db",
    "hybrid search", "ann", "approximate nearest neighbour",
    "information retrieval", "ranking", "bm25", "learning-to-rank",
    "recommendation systems", "recommendation", "search systems",
    "retrieval augmented", "rag", "reranking", "re-ranking", "colbert",
    "dense passage retrieval", "nlp", "natural language processing",
    "text classification", "named entity recognition", "ner",
    "question answering", "text mining", "llm", "large language model",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft", "transformer",
    "bert", "gpt", "language model", "hugging face", "huggingface",
    "llm fine-tuning", "ndcg", "mrr", "map", "ranking evaluation",
    "a/b testing", "offline evaluation", "retrieval evaluation", "python",
})
IMPORTANT_SKILLS = frozenset({
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "mlops",
    "mlflow", "weights & biases", "wandb", "experiment tracking", "docker",
    "kubernetes", "spark", "kafka", "airflow", "dbt", "feature engineering",
    "model serving", "triton", "ray", "distributed systems", "sql", "git",
    "github", "redis", "model deployment", "inference optimization", "streaming",
})
WRONG_DOMAIN = frozenset({
    "computer vision", "image classification", "object detection", "cnn",
    "yolo", "image segmentation", "convolutional", "speech recognition",
    "text-to-speech", "tts", "asr", "automatic speech recognition",
    "robotics", "ros",
})

# Career-description keywords → production ML/IR evidence
ML_PRODUCTION_KW = (
    "embedding", "vector", "retrieval", "search", "ranking", "recommendation",
    "nlp", "language model", "transformer", "bert", "rag", "fine-tun",
    "semantic", "rerank", "a/b test", "index", "production", "deployed",
    "scale", "latency",
)
# Stronger "built retrieval/ranking/recsys" evidence for the role_gate lift
RETRIEVAL_BUILD_KW = (
    "retrieval", "ranking", "recommendation", "recsys", "embedding",
    "vector", "semantic search", "search relevance", "rerank",
)

CONSULTING_FIRMS = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl ", "tech mahindra", "l&t infotech", "mphasis",
    "hexaware", "niit tech", "ibm consulting", "kyndryl",
})
PREF_CITIES = frozenset({
    "pune", "noida", "hyderabad", "mumbai", "bangalore", "bengaluru",
    "delhi", "gurugram", "gurgaon", "chennai", "kolkata", "new delhi",
})

# Academic / research signal (for academic_only_no_prod penalty)
ACADEMIC_KW = ("phd", "postdoc", "research fellow", "research assistant",
               "university", "institute", "publication", "paper")
PROD_KW = ("production", "deployed", "shipped", "serving", "scale", "latency",
           "real users", "a/b test", "pipeline")
LANGCHAIN_KW = ("langchain", "llamaindex", "llama-index")
