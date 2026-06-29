import streamlit as st
import sys
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – RAG Explorer",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e2e8f0;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* App header */
.app-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 2rem;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 1rem;
}
.app-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem;
    font-weight: 600;
    color: #7dd3fc;
    letter-spacing: -0.5px;
}
.app-sub {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 300;
}

/* Section labels */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
}

/* Answer box */
.answer-box {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #7dd3fc;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #e2e8f0;
    white-space: pre-wrap;
    margin-bottom: 1.5rem;
}

/* Score card */
.score-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.score-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.score-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.score-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
}
.score-high  { color: #4ade80; }
.score-mid   { color: #facc15; }
.score-low   { color: #f87171; }

/* Stats row */
.stats-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: #1e293b;
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #94a3b8;
    display: flex;
    align-items: center;
    gap: 6px;
}
.stat-pill span.val {
    color: #e2e8f0;
    font-weight: 600;
}

/* Verdict badge */
.verdict-pass {
    display: inline-block;
    background: #052e16;
    border: 1px solid #166534;
    color: #4ade80;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
}
.verdict-fail {
    display: inline-block;
    background: #2d0a0a;
    border: 1px solid #7f1d1d;
    color: #f87171;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
}
.verdict-notfound {
    display: inline-block;
    background: #1c1917;
    border: 1px solid #44403c;
    color: #a8a29e;
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
}

/* Run button */
.stButton > button {
    background: #1d4ed8;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.5rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    width: 100%;
    cursor: pointer;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #2563eb;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111827;
    border: 1px dashed #334155;
    border-radius: 8px;
    padding: 0.5rem;
}

/* Divider */
.thin-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 1.25rem 0;
}

/* Not-found box */
.notfound-box {
    background: #1c1917;
    border: 1px solid #44403c;
    border-left: 3px solid #a8a29e;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    color: #a8a29e;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <span class="app-title">DocMind</span>
    <span class="app-sub">RAG · Retrieve · Evaluate</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar – file upload ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Upload Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="",
        type=["pdf", "csv", "png", "jpg", "jpeg", "bmp"],
        help="Supported: PDF (text or scanned), CSV, Images"
    )

    st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#64748b; line-height:1.6;">
    DocMind runs a full agentic RAG pipeline:<br><br>
    <b style="color:#94a3b8">1.</b> Detects file type<br>
    <b style="color:#94a3b8">2.</b> Loads & chunks document<br>
    <b style="color:#94a3b8">3.</b> Embeds into FAISS<br>
    <b style="color:#94a3b8">4.</b> Retrieves & grades chunks<br>
    <b style="color:#94a3b8">5.</b> Generates answer<br>
    <b style="color:#94a3b8">6.</b> Critic re-evaluates<br>
    <b style="color:#94a3b8">7.</b> Scores with 4 metrics
    </div>
    """, unsafe_allow_html=True)


# ── Main area ─────────────────────────────────────────────────────────────────
col_q, col_run = st.columns([5, 1])

with col_q:
    question = st.text_input(
        label="",
        placeholder="Ask anything about your document…",
        label_visibility="collapsed"
    )

with col_run:
    run_btn = st.button("→ Ask", use_container_width=True)


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not uploaded_file:
        st.warning("Please upload a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        # Save upload to a temp file
        import tempfile
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Running pipeline… this may take a moment"):
            try:
                # Import the compiled graph from main.py
                # Make sure main.py is in the same folder as frontend.py
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from main import graph

                result = graph.invoke({
                    "file_path": tmp_path,
                    "question": question.strip()
                })

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        # ── Pull values from result ───────────────────────────────────────────
        answer         = result.get("answer", "No answer returned.")
        scores         = result.get("ragas_scores", {})
        critic_verdict = result.get("critic_verdict", "unknown")
        critic_reason  = result.get("critic_reason", "")
        ret_attempts   = result.get("retrieval_attempts", 0)
        crit_attempts  = result.get("critic_attempts", 0)
        final_verdict  = result.get("final_verdict", "")

        # ── Stats pills ───────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Pipeline Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-pill">Retrieval attempts <span class="val">{ret_attempts}</span></div>
            <div class="stat-pill">Critic attempts <span class="val">{crit_attempts}</span></div>
            <div class="stat-pill">Critic verdict
                <span class="val" style="color:{'#4ade80' if critic_verdict=='pass' else '#f87171'}">
                    {critic_verdict.upper()}
                </span>
            </div>
            {'<div class="stat-pill">Reason <span class="val">' + critic_reason + '</span></div>' if critic_reason else ''}
        </div>
        """, unsafe_allow_html=True)

        # ── Answer ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)

        if final_verdict == "answer_not_found":
            st.markdown(f'<div class="notfound-box">⚠ {answer}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        # ── Verdict badge ─────────────────────────────────────────────────────
        if final_verdict == "answer_not_found":
            st.markdown('<span class="verdict-notfound">NOT FOUND</span>', unsafe_allow_html=True)
        elif critic_verdict == "pass":
            st.markdown('<span class="verdict-pass">✓ VERIFIED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="verdict-fail">✗ UNVERIFIED</span>', unsafe_allow_html=True)

        # ── Evaluation scores ─────────────────────────────────────────────────
        if scores and final_verdict != "answer_not_found":
            st.markdown('<hr class="thin-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Evaluation Scores</div>', unsafe_allow_html=True)

            metric_labels = {
                "faithfulness":      "Faithfulness",
                "answer_relevancy":  "Answer Relevancy",
                "context_precision": "Context Precision",
                "context_recall":    "Context Recall",
            }

            cards_html = '<div class="score-grid">'
            for key, label in metric_labels.items():
                val = scores.get(key, None)
                if val is None:
                    continue
                try:
                    val_f = float(val)
                except Exception:
                    val_f = 0.0

                css_class = (
                    "score-high" if val_f >= 0.75 else
                    "score-mid"  if val_f >= 0.45 else
                    "score-low"
                )
                bar_pct = int(val_f * 100)
                bar_color = (
                    "#4ade80" if val_f >= 0.75 else
                    "#facc15" if val_f >= 0.45 else
                    "#f87171"
                )

                cards_html += f"""
                <div class="score-card">
                    <div class="score-label">{label}</div>
                    <div class="score-value {css_class}">{val_f:.2f}</div>
                    <div style="margin-top:8px; background:#1e293b; border-radius:4px; height:4px; overflow:hidden;">
                        <div style="width:{bar_pct}%; background:{bar_color}; height:4px; border-radius:4px;"></div>
                    </div>
                </div>
                """

            cards_html += "</div>"

            if "error" in scores:
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#f87171;margin-top:0.5rem;">⚠ Score parsing error: {scores["error"]}</div>',
                    unsafe_allow_html=True
                )

            st.markdown(cards_html, unsafe_allow_html=True)

# ── Empty state ───────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 0; color:#334155;">
        <div style="font-size:2.5rem; margin-bottom:1rem;">🔍</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.85rem; letter-spacing:2px; text-transform:uppercase;">
            Upload a file and ask a question
        </div>
    </div>
    """, unsafe_allow_html=True)