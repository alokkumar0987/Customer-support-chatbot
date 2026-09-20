# System Architecture Diagram

## 🏗️ **Complete System Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIVER TAKE-HOME SOLUTION                      │
│                   AI Customer Support Agent                       │
│                        (@AmazonHelp)                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Raw Dataset (TWCS):                                             │
│  └─ 2.8M tweets → AmazonHelp conversations                      │
│                                                                   │
│  Conversation Reconstruction (DSU):                              │
│  └─ Graph-based clustering → unique conversations               │
│                                                                   │
│  Conversation-Level Split (Zero Leakage):                        │
│  ├─ Training:   122,209 pairs (retrieval corpus, English-only)  │
│  ├─ Gold Test:  200 examples (official evaluation set)          │
│  └─ Verified:   train ∩ test = ∅ (tests/test_leakage.py)       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BASELINE SYSTEMS (B0, B1, B2)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  B0: Majority Classifier (Trivial Baseline)                     │
│  └─ Always predicts most common intent                          │
│   |__ Intent F1: ~3%                                               │
│                                                                   │
│  B1: LLM-Pseudo-Labeled TF-IDF + Logistic Regression            │
│  ├─ Uses pseudo-labels from DeepSeek/LLM (NOT human-annotated)  │
│  ├─ Intent F1: ~70% (from benchmark.json)                       │
│  └─ Escalation F1: ~73%                                          │
│                                                                   │
│  B2: LLM Zero-Shot (Simple LLM)                                  │
│  ├─ Gemini 2.5 Flash with intent taxonomy                       │
│  ├─ Intent F1: ~79% (from benchmark.json)                       │
│  └─ Escalation F1: ~76%                                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PROPOSED SYSTEM (LangGraph + HITL)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│                    ┌──────────────┐                              │
│                    │    START     │                              │
│                    └──────┬───────┘                              │
│                           ↓                                       │
│                  ┌─────────────────┐                             │
│                  │ classify_intent │ ← LLM Intent Classifier     │
│                  │  (6 intents)    │   + Confidence Score        │
│                  └────────┬────────┘                             │
│                           ↓                                       │
│                  ┌─────────────────┐                             │
│                  │  detect_risk    │ ← Regex Pattern Matching   │
│                  │ (hard + soft)   │   Legal, Fraud, Human, etc.│
│                  └────────┬────────┘                             │
│                           ↓                                       │
│            ┌──────────────────────────────┐                      │
│            │     EARLY RISK GATE          │                      │
│            │  (Hard Risks: Legal, Fraud,  │                      │
│            │   Explicit Human Request)    │                      │
│            └──────────────┬───────────────┘                      │
│                    ╱              ╲                               │
│              HARD RISK           NORMAL                           │
│                  ↓                  ↓                             │
│         ┌────────────────┐  ┌──────────────┐                    │
│         │escalate_high_  │  │   retrieve   │ ← Hybrid RAG           │
│         │   risk         │  │ (top-5 cases)│   BM25+Dense+RRF       │
│         └────────┬───────┘  └──────┬───────┘                    │
│                  │                  ↓                             │
│                  │         ┌─────────────────┐                   │
│                  │         │     draft       │ ← RAG Generation  │
│                  │         │  (grounded)     │   + Evidence      │
│                  │         └────────┬────────┘                   │
│                  │                  ↓                             │
│                  │         ┌─────────────────┐                   │
│                  │         │    verify       │ ← LLM Verifier    │
│                  │         │ (fail-closed)   │   Groundedness    │
│                  │         └────────┬────────┘                   │
│                  │                  ↓                             │
│                  │    ┌──────────────────────────────┐           │
│                  │    │    SAFETY GATE               │           │
│                  │    │ (Verification + Confidence   │           │
│                  │    │  + Evidence + Soft Risks)    │           │
│                  │    └──────────┬───────────────────┘           │
│                  │         ╱            ╲                         │
│                  │    UNSAFE          SAFE                        │
│                  │       ↓               ↓                        │
│                  │  ┌─────────┐  ┌──────────────┐               │
│                  │  │escalate_│  │ auto_handle  │               │
│                  │  │uncertain│  └──────┬───────┘               │
│                  │  └────┬────┘         ↓                        │
│                  │       ↓             END                        │
│                  ↓       ↓              ↑                         │
│            ┌──────────────────┐         │                        │
│            │  human_review    │         │                        │
│            │ interrupt() HITL │         │                        │
│            └────────┬─────────┘         │                        │
│                     ↓                    │                        │
│         ┌───────────────────────┐       │                        │
│         │ Human Actions:        │       │                        │
│         │ [A] Approve           │       │                        │
│         │ [E] Edit              │───────┘                        │
│         │ [R] Replace           │                                │
│         └───────────────────────┘                                │
│                                                                   │
│  Key Metrics (from benchmark.json, 200-example evaluation):     │
│  ├─ Intent F1: 93.6%                                             │
│  ├─ Escalation F1: 95.2%                                         │
│  └─ Unsafe FNR: 0.0% (0/18 balanced, 0/14 natural)              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      EVALUATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Gold-Standard Evaluation:                                       │
│  ├─ 200 held-out examples (no leakage)                          │
│  ├─ Intent: Accuracy, P/R/F1                                    │
│  ├─ Escalation: Precision, Recall, F1, Unsafe FNR              │
│  └─ Response Quality: LLM Judge (validated: κ=0.88, 50 samples) │
│                                                                   │
│  Baseline Comparison:                                            │
│  └─ B0 vs B1 vs B2 vs Proposed (head-to-head)                  │
│                                                                   │
│  Ablation Study:                                                 │
│  └─ LLM → +Retrieval → +Verifier → +Risk Detection             │
│                                                                   │
│  Adversarial Testing:                                            │
│  └─ 15 red-team cases (legal, fraud, injection, etc.)          │
│                                                                   │
│  HITL Metrics (Note: actual HITL run needed for real data):     │
│  └─ TBD - requires running interactive HITL sessions            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 **Component Deep Dive**

### **Node Descriptions**

| Node | Input | Output | Purpose |
|------|-------|--------|---------|
| `classify_intent` | Customer message | Intent + confidence | LLM classification (6 classes) |
| `detect_risk` | Customer message | Risk attributes | Pattern matching (legal, fraud, etc.) |
| `escalate_high_risk` | Risks | Decision + reason | Mark hard risks for HITL |
| `retrieve` | Customer message | Top-5 cases + score | Hybrid retrieval (BM25+Dense+Rerank) |
| `draft` | Message + evidence | Draft response | RAG-based generation |
| `verify` | Draft + evidence | Verification result | Groundedness check (fail-closed) |
| `escalate_uncertain` | Verification | Decision + reason | Mark unsafe for HITL |
| `human_review` | All context | Human feedback | HITL with `interrupt()` |
| `auto_handle` | Draft | Final response | Finalize auto-handled cases |

### **Hybrid Retrieval Pipeline**

The 
etrieve node implements a production-grade hybrid RAG pipeline:

\customer_query
    ↓
query_normalizer
    → remove URLs (@mentions)
    → normalize punctuation  
    → clean whitespace
    ↓
hybrid_retriever
    ├── BM25Okapi (sparse, top 50)
    └── Dense/FAISS (BGE embeddings, top 50)
    ↓
RRF fusion (k=60)
    ↓
deduplication
    ↓
top 50 candidates
    ↓
cross_encoder_reranker (MS-MARCO MiniLM)
    ↓
top 5 resolution cases
    ↓
context_builder
    → structured evidence
    → grounding rules
    → provenance tracking
    ↓
LLM generation
\
**Key Design Principles:**
- **Retrieval Independence**: Retrieval does NOT filter by predicted intent (intent errors don't contaminate evidence)
- **No Arbitrary Chunking**: Retrieval unit = complete customer→brand resolution case (preserves full context)
- **Evidence as Patterns**: Historical replies are resolution patterns, NOT current account/order/refund truth
- **Fail-Closed Verification**: Exceptions → unsafe → escalate (never assumes safe on error)
- **Leakage Protection**: Index builder validates conversation ID overlap and FAILS LOUDLY if detected

**Models:**
- Dense Embeddings: BAAI/bge-small-en-v1.5 (384-dim)
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Vector Store: FAISS IndexFlatIP (exact search)

**To build indexes:** \python scripts/build_rag_index.py\ (requires leakage-free corpus)

### **Routing Logic**

**Early Gate (`route_after_risk_detection`):**
```python
if any(hard_risk):
    return "escalate_high_risk"  # → HITL (no drafting)
else:
    return "retrieve"  # → Continue normal flow
```

**Safety Gate (`route_after_verification`):**
```python
unsafe = [
    not grounded,
    unsupported_claim,
    contradiction,
    low_confidence,
    weak_evidence,
    soft_risk_present
]

if any(unsafe):
    return "escalate_uncertain"  # → HITL
else:
    return "auto_handle"  # → AUTO
```

### **Risk Taxonomy**

**Hard Risks (Early Escalation):**
- `legal_threat` - "I'll sue", "lawyer", "court"
- `fraud_financial_risk` - "fraud", "stolen", "unauthorized"
- `explicit_human_agent_request` - "speak to human", "real person"

**Soft Risks (Safety Gate):**
- `pii_account_verification` - "password", "account number"
- `severe_negative_sentiment` - "terrible", "worst", "hate"
- `policy_exception_request` - "exception", "waive", "override"

### **Intent Taxonomy**

```
1. order_shipping_tracking      - Order/package location
2. refund_payment_inquiry        - Refund status/timing
3. return_exchange_request       - Product return/exchange
4. account_digital_support       - Login/account access
5. order_cancellation_change     - Cancel/modify order
6. general_product_inquiry       - Product info/availability
```

---

## 🎯 **Decision Flow Examples**

### **Example 1: Normal Query → Auto-Handle**

```
Input: "Where is my order? I ordered 3 days ago."

classify_intent → "order_shipping_tracking" (confidence: 0.95)
detect_risk → []
Early Gate → retrieve (no hard risks)
retrieve → 5 cases (top score: 0.61, sufficient)
draft → "Hi! Please check tracking at..."
verify → grounded: true, unsupported: false
Safety Gate → auto_handle (all checks pass)

Decision: AUTO_HANDLE
Reason: None (handled successfully)
```

### **Example 2: Legal Threat → Early Escalation**

```
Input: "I will sue Amazon if you don't refund me!"

classify_intent → "refund_payment_inquiry" (0.95)
detect_risk → ["legal_threat", "severe_negative_sentiment"]
Early Gate → escalate_high_risk (hard risk detected)

Decision: ESCALATE_TO_HUMAN
Reason: "Hard risk: legal_threat"
Draft: None (not generated)
→ interrupt() → Human Review
```

### **Example 3: Weak Evidence → Safety Gate Escalation**

```
Input: "My order never arrived."

classify_intent → "order_shipping_tracking" (0.88)
detect_risk → []
Early Gate → retrieve
retrieve → 5 cases (top score: 0.12, insufficient!)
draft → "Please provide order number..."
verify → grounded: false, insufficient_evidence: true
Safety Gate → escalate_uncertain

Decision: ESCALATE_TO_HUMAN
Reason: "Weak retrieval (score: 0.12); Insufficient evidence"
→ interrupt() → Human Review
```

---

## 📊 **Data Flow**

```
Twitter Dataset (2.8M tweets)
         ↓
DSU Conversation Reconstruction
         ↓
AmazonHelp Filter
         ↓
Conversation-Level Split
    ├─ Training: 122,209 (retrieval corpus, English-only)
    └─ Gold: 200 (held-out test)
         ↓
┌────────────────┬─────────────────┐
Training         Gold Test         
   ↓                 ↓
Build           Evaluate
Retrieval       B0, B1, B2
Index           Proposed
   ↓                 ↓
Used by         Performance
Proposed        Comparison
System              ↓
                 Results
```

---

## 🔐 **Safety Guarantees**

1. **Fail-Closed Verification**
   - Exception → Unsafe → Escalate
   - Never assumes safe on error

2. **Zero Leakage**
   - Training ∩ Test = ∅
   - Verified by automated tests

3. **Deterministic Routing**
   - No LLM for escalation decisions
   - Auditable if/else logic

4. **Stated Reasons**
   - Every escalation has explanation
   - Transparent decision-making

5. **HITL Guardrail**
   - Uncertain cases reviewed by human
   - Approval/edit/replace options

---

## 📁 **Code Organization**

```
project/
├── src/                     # Reusable ML components
│   ├── intent_classifier.py
│   ├── retrieval.py
│   ├── reply_generator.py
│   └── escalation_router.py
│
├── app/                     # LangGraph orchestration
│   ├── graph/
│   │   ├── graph.py        # Graph construction
│   │   ├── nodes.py        # 9 nodes
│   │   ├── routing.py      # Deterministic routing
│   │   └── state.py        # State schema
│   └── safety/
│       ├── risk_detector.py
│       └── response_verifier.py
│
├── eval/                    # Evaluation suite
│   ├── run_evaluation.py
│   ├── ablation.py
│   ├── llm_judge.py
│   └── judge_agreement.py
│
├── tests/                   # Test suite
│   ├── test_leakage.py
│   ├── test_adversarial.py
│   └── test_agent.py
│
├── data/
│   ├── processed/
│   │   └── amazonhelp_retrieval_corpus.json
│   └── golden_eval_set_balanced.json
│
└── results/
    ├── benchmark.json
    ├── ablation.json
    └── judge_agreement.json
```

---

**This architecture satisfies all Hiver requirements with systematic evidence.** ✅

