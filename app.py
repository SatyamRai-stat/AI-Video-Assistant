import streamlit as st
from dotenv import load_dotenv
import sys
import os
import time

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_processor import process_input
from core.transcribe import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

st.set_page_config(
    page_title="InsightAI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: #07090f !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #94a3b8 !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stMain"] > div { padding-top: 0 !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 4px; }

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 90% 60% at 50% -5%, rgba(99,102,241,0.13) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 5%  70%, rgba(16,185,129,0.05) 0%, transparent 50%),
        radial-gradient(ellipse 40% 35% at 95% 80%, rgba(139,92,246,0.07) 0%, transparent 50%);
    pointer-events: none;
}
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px);
    background-size: 56px 56px;
    pointer-events: none;
}

/* ── root ── */
.ia-root { position: relative; z-index: 2; }

/* ══════════════════
   HERO
══════════════════ */
.ia-hero {
    padding: 80px 24px 52px;
    text-align: center;
    max-width: 640px;
    margin: 0 auto;
}
.ia-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 24px;
    padding: 6px 16px 6px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
    color: #818cf8; margin-bottom: 32px;
}
.ia-badge-dot {
    width: 6px; height: 6px;
    background: #818cf8; border-radius: 50%;
    box-shadow: 0 0 8px #818cf8;
    animation: bdot 2.2s ease-in-out infinite;
}
@keyframes bdot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.7)} }

.ia-hero-title {
    font-size: clamp(36px, 5vw, 64px);
    font-weight: 700; line-height: 1.06;
    letter-spacing: -0.03em; color: #f1f5f9;
    margin: 0 0 18px;
}
.ia-hero-title em {
    font-style: normal;
    background: linear-gradient(120deg, #818cf8 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ia-hero-sub {
    font-size: 15px; color: #3d5570;
    line-height: 1.75; max-width: 380px;
    margin: 0 auto 52px;
}

/* ══════════════════
   PANEL SHELL
══════════════════ */
.ia-panel-outer {
    width: min(560px, 96vw);
    margin: 0 auto;
    background: rgba(13,19,34,0.88);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px;
    padding: 24px 24px 28px;
    backdrop-filter: blur(28px);
    box-shadow:
        0 0 0 1px rgba(99,102,241,0.05),
        0 28px 72px rgba(0,0,0,0.55);
}

/* ── Source toggle: two styled Streamlit buttons ── */
/* We use col buttons and override them per class */

/* YouTube tab button */
div[data-testid="stButton"].yt-btn button {
    background: rgba(99,102,241,0.1) !important;
    border: 1.5px solid rgba(99,102,241,0.45) !important;
    border-radius: 14px !important;
    color: #a5b4fc !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 14px 12px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.12), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}

/* Audio tab button */
div[data-testid="stButton"].aud-btn button {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    color: #475569 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 14px 12px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

/* When audio is active */
div[data-testid="stButton"].aud-btn-active button {
    background: rgba(99,102,241,0.1) !important;
    border: 1.5px solid rgba(99,102,241,0.45) !important;
    border-radius: 14px !important;
    color: #a5b4fc !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 14px 12px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.12), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}
div[data-testid="stButton"].yt-btn-inactive button {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    color: #475569 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 14px 12px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

/* hover for inactive tabs */
div[data-testid="stButton"].aud-btn button:hover,
div[data-testid="stButton"].yt-btn-inactive button:hover {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.14) !important;
    color: #94a3b8 !important;
}

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    color: #c8d4e4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    padding: 13px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s, background 0.2s !important;
    caret-color: #818cf8 !important;
    width: 100% !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(99,102,241,0.45) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.08) !important;
    background: rgba(99,102,241,0.04) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder { color: #1e3a5a !important; }
[data-testid="stTextInput"] label { display: none !important; }
[data-testid="stTextInput"] { margin-top: 6px !important; }

/* ── Analyze button ── */
div[data-testid="stButton"].analyze-btn button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 13px 28px !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.28), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    width: 100% !important;
    margin-top: 8px !important;
}
div[data-testid="stButton"].analyze-btn button:hover {
    background: linear-gradient(135deg, #7c7ff5 0%, #5d56ee 100%) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.38), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"].analyze-btn button:active { transform: translateY(0) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    transition: all 0.2s !important;
    margin-top: 6px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(99,102,241,0.3) !important;
    background: rgba(99,102,241,0.04) !important;
}
[data-testid="stFileUploader"] label { color: #334155 !important; font-size: 13px !important; }
[data-testid="stFileUploader"] { margin-top: 2px !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #34d399) !important;
    border-radius: 4px !important;
    transition: width 0.4s ease !important;
}
[data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 4px !important; height: 3px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(13,19,34,0.6) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important; overflow: hidden !important;
    max-width: 1080px; margin: 0 auto 32px;
}
[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; color: #2d4460 !important;
    padding: 14px 20px !important; letter-spacing: 0.05em !important;
}
[data-testid="stExpander"] summary:hover { color: #64748b !important; }
[data-testid="stExpander"] summary svg { fill: #2d4460 !important; }

/* ── Chat ── */
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; margin-bottom: 12px !important; }
[data-testid="stChatMessageContent"] {
    background: rgba(13,19,34,0.7) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    font-size: 14px !important; color: #64748b !important; line-height: 1.75 !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: rgba(99,102,241,0.06) !important;
    border-color: rgba(99,102,241,0.12) !important;
    color: #a5b4fc !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(13,19,34,0.9) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #cbd5e1 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    caret-color: #818cf8 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99,102,241,0.35) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.06) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #1e3a5a !important; }

/* ── Alert ── */
[data-testid="stAlert"] {
    background: rgba(239,68,68,0.06) !important;
    border: 1px solid rgba(239,68,68,0.15) !important;
    border-radius: 10px !important; color: #f87171 !important;
}
[data-testid="stWarningAlertContent"] { color: #fbbf24 !important; }

/* ═══════════════
   DIVIDER
═══════════════ */
.ia-divider {
    width: 100%; height: 1px; border: none;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.2) 30%, rgba(52,211,153,0.14) 70%, transparent);
    margin: 60px 0; position: relative;
}
.ia-divider::after {
    content: '✦';
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%,-50%);
    background: #07090f; color: #2a3a50;
    font-size: 11px; padding: 0 14px;
}

/* ═══════════════
   RESULTS
═══════════════ */
.ia-result-header { text-align: center; padding: 0 24px 40px; }
.ia-result-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
    color: #34d399; margin-bottom: 12px;
    display: flex; align-items: center; justify-content: center; gap: 8px;
}
.ia-result-eyebrow::before, .ia-result-eyebrow::after {
    content: ''; display: inline-block;
    width: 28px; height: 1px; background: rgba(52,211,153,0.25);
}
.ia-result-title {
    font-size: clamp(22px, 3.2vw, 42px);
    font-weight: 600; letter-spacing: -0.025em;
    line-height: 1.15; color: #e2e8f0;
}

.ia-stats {
    display: flex; justify-content: center;
    gap: 10px; flex-wrap: wrap;
    padding: 0 24px; margin-bottom: 40px;
}
.ia-stat {
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 5px 14px;
    color: #2d4460;
    display: flex; align-items: center; gap: 6px;
}
.ia-stat-val { color: #818cf8; font-weight: 500; }
.ia-stat-val.ok { color: #34d399; }

/* ── Cards ── */
.ia-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 440px), 1fr));
    gap: 16px; padding: 0 24px;
    max-width: 1100px; margin: 0 auto 28px;
}
.ia-card {
    background: rgba(13,19,34,0.72);
    border-radius: 16px; padding: 24px 26px;
    backdrop-filter: blur(16px);
    border: 1px solid var(--c-border);
    position: relative; overflow: hidden;
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}
.ia-card::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: var(--c-top-line); opacity: 0.6;
}
.ia-card:hover { transform: translateY(-3px); border-color: var(--c-border-hover); box-shadow: 0 20px 48px rgba(0,0,0,0.3); }
.ia-card-head { display: flex; align-items: center; gap: 11px; margin-bottom: 16px; }
.ia-card-icon {
    width: 34px; height: 34px; border-radius: 8px;
    background: var(--c-icon-bg); color: var(--c-icon);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-family: 'IBM Plex Mono', monospace; font-weight: 500; flex-shrink: 0;
}
.ia-card-label { font-size: 11px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--c-label); }
.ia-card-content { font-size: 14px; line-height: 1.8; color: #3d556a; word-break: break-word; }
.ia-card-content p { margin: 0; }
.ia-card-content ul { margin: 0; padding-left: 16px; list-style: none; }
.ia-card-content ul li { position: relative; padding-left: 14px; margin-bottom: 7px; color: #3d556a; }
.ia-card-content ul li::before { content: '—'; position: absolute; left: 0; color: var(--c-icon); opacity: 0.5; font-size: 12px; }
.ia-card-indigo { --c-border:rgba(99,102,241,0.12); --c-border-hover:rgba(99,102,241,0.28); --c-top-line:linear-gradient(90deg,rgba(99,102,241,0.6),transparent); --c-icon-bg:rgba(99,102,241,0.1); --c-icon:#818cf8; --c-label:#6366f1; }
.ia-card-teal   { --c-border:rgba(20,184,166,0.12); --c-border-hover:rgba(20,184,166,0.28); --c-top-line:linear-gradient(90deg,rgba(20,184,166,0.6),transparent); --c-icon-bg:rgba(20,184,166,0.1); --c-icon:#2dd4bf; --c-label:#14b8a6; }
.ia-card-amber  { --c-border:rgba(245,158,11,0.12); --c-border-hover:rgba(245,158,11,0.28); --c-top-line:linear-gradient(90deg,rgba(245,158,11,0.6),transparent); --c-icon-bg:rgba(245,158,11,0.08); --c-icon:#fbbf24; --c-label:#d97706; }
.ia-card-violet { --c-border:rgba(139,92,246,0.12); --c-border-hover:rgba(139,92,246,0.28); --c-top-line:linear-gradient(90deg,rgba(139,92,246,0.6),transparent); --c-icon-bg:rgba(139,92,246,0.1); --c-icon:#a78bfa; --c-label:#8b5cf6; }

/* ── Processing ── */
.ia-proc-wrap { display: flex; flex-direction: column; align-items: center; padding: 64px 24px; }
.ia-proc-card {
    width: min(480px, 95vw);
    background: rgba(13,19,34,0.92);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 40px 36px; text-align: center;
    backdrop-filter: blur(24px);
    box-shadow: 0 32px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.05);
}
.ia-proc-icon {
    width: 58px; height: 58px; border-radius: 16px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.2);
    margin: 0 auto 18px;
    display: flex; align-items: center; justify-content: center; font-size: 24px;
    animation: iconFloat 3s ease-in-out infinite;
}
@keyframes iconFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-4px)} }
.ia-proc-title { font-size: 20px; font-weight: 600; color: #e2e8f0; margin-bottom: 4px; }
.ia-proc-sub { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #1e3a5a; letter-spacing: 0.08em; }
.ia-steps { margin-top: 26px; text-align: left; display: flex; flex-direction: column; gap: 5px; }
.ia-step { display: flex; align-items: center; gap: 12px; padding: 9px 13px; border-radius: 8px; font-size: 13px; font-family: 'IBM Plex Mono', monospace; }
.ia-step-done   { color: #34d399; background: rgba(52,211,153,0.05); }
.ia-step-active { color: #818cf8; background: rgba(99,102,241,0.07); animation: stepBlink 1.4s ease infinite; }
.ia-step-idle   { color: #1e3a5a; }
@keyframes stepBlink { 0%,100%{opacity:1} 50%{opacity:0.45} }
.ia-step-pip { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.ia-step-done   .ia-step-pip { background: #34d399; }
.ia-step-active .ia-step-pip { background: #818cf8; box-shadow: 0 0 8px #818cf8; }
.ia-step-idle   .ia-step-pip { background: #1a2640; border: 1px solid #1e3a5a; }

/* ── Chat ── */
.ia-chat-header {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 22px;
    background: rgba(13,19,34,0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; margin-bottom: 20px;
    max-width: 860px; margin-left: auto; margin-right: auto;
}
.ia-chat-avatar {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(52,211,153,0.1));
    border: 1px solid rgba(99,102,241,0.15);
    display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0;
}
.ia-chat-title { font-size: 15px; font-weight: 600; color: #cbd5e1; margin-bottom: 2px; }
.ia-chat-desc { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #1e3a5a; letter-spacing: 0.05em; }
.ia-chat-wrap { max-width: 860px; margin: 0 auto; padding: 0 24px 80px; }

/* ── Features (empty state) ── */
.ia-features {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 14px; max-width: 1000px; margin: 0 auto; padding: 0 24px;
}
.ia-feat {
    background: rgba(13,19,34,0.5);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 14px; padding: 24px;
    transition: border-color 0.2s, transform 0.2s;
    position: relative; overflow: hidden;
}
.ia-feat::after {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: var(--feat-accent, rgba(99,102,241,0.4));
    opacity: 0; transition: opacity 0.2s;
}
.ia-feat:hover { border-color: rgba(255,255,255,0.08); transform: translateY(-2px); }
.ia-feat:hover::after { opacity: 1; }
.ia-feat-glyph { font-size: 22px; margin-bottom: 14px; display: block; }
.ia-feat-name  { font-size: 14px; font-weight: 600; color: #94a3b8; margin-bottom: 6px; }
.ia-feat-desc  { font-size: 13px; color: #1e3a5a; line-height: 1.65; }
.ia-section-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
    color: #1e3a5a; text-align: center; margin-bottom: 28px;
}

/* ── Tab styling helpers via container key ── */
div[data-testid="column"]:has(div.tab-yt-active) button {
    background: rgba(99,102,241,0.1) !important;
    border: 1.5px solid rgba(99,102,241,0.5) !important;
    color: #a5b4fc !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.15) !important;
}
div[data-testid="column"]:has(div.tab-inactive) button {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #3d5570 !important;
}
div[data-testid="column"]:has(div.tab-inactive) button:hover {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.14) !important;
    color: #64748b !important;
}

/* universal tab button base */
div[data-testid="stButton"] button {
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 14px 12px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    letter-spacing: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "pipeline_result": None,
        "chat_history": [],
        "error": None,
        "current_source": None,
        "input_mode": "youtube",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="ia-root">', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ia-hero">
  <div class="ia-badge">
    <div class="ia-badge-dot"></div>
    InsightAI &nbsp;·&nbsp; v2.0
  </div>
  <h1 class="ia-hero-title">
    Turn audio into<br><em>structured insight</em>
  </h1>
  <p class="ia-hero-sub">
    Drop a YouTube link or audio file. Get a full transcript,
    AI summary, action items, and a chat interface — in under a minute.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Input panel ───────────────────────────────────────────────────────────────
_, panel_col, _ = st.columns([1, 2.0, 1])

with panel_col:
    mode = st.session_state.input_mode

    # Thin top-border glow on panel container
    st.markdown("""
    <style>
    div[data-testid="stVerticalBlock"]:has(div.ia-panel-marker) {
        background: rgba(13,19,34,0.88);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 22px;
        padding: 20px !important;
        backdrop-filter: blur(28px);
        box-shadow: 0 0 0 1px rgba(99,102,241,0.05), 0 28px 72px rgba(0,0,0,0.55);
        position: relative;
    }
    div[data-testid="stVerticalBlock"]:has(div.ia-panel-marker)::before {
        content: '';
        position: absolute;
        top: 0; left: 10%; right: 10%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent);
        border-radius: 99px;
    }
    </style>
    <div class="ia-panel-marker"></div>
    """, unsafe_allow_html=True)

    # ── Tab buttons ────────────────────────────────────────────────────────────
    t1, t2 = st.columns(2, gap="small")

    with t1:
        yt_class = "tab-yt-active" if mode == "youtube" else "tab-inactive"
        st.markdown(f'<div class="{yt_class}"></div>', unsafe_allow_html=True)
        yt_label = "▶  YouTube URL" if mode == "youtube" else "▶  YouTube URL"
        if st.button(yt_label, key="tab_yt", use_container_width=True):
            st.session_state.input_mode = "youtube"
            st.rerun()

    with t2:
        aud_class = "tab-yt-active" if mode == "audio" else "tab-inactive"
        st.markdown(f'<div class="{aud_class}"></div>', unsafe_allow_html=True)
        if st.button("♪  Audio File", key="tab_aud", use_container_width=True):
            st.session_state.input_mode = "audio"
            st.rerun()

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Input fields ───────────────────────────────────────────────────────────
    source_path: str | None = None

    if mode == "youtube":
        url_val = st.text_input(
            "url",
            placeholder="https://youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="url_input",
        )
        st.markdown("""
        <style>
        div[data-testid="stButton"]:has(button[data-testid="stBaseButton-secondary"]) button {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
            border: none !important;
            color: #fff !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
            margin-top: 4px !important;
        }
        div[data-testid="stButton"]:has(button[data-testid="stBaseButton-secondary"]) button:hover {
            background: linear-gradient(135deg, #7c7ff5 0%, #5d56ee 100%) !important;
            box-shadow: 0 8px 32px rgba(99,102,241,0.4) !important;
            transform: translateY(-1px) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button("Analyze video  →", use_container_width=True, key="btn_yt"):
            if not url_val.strip():
                st.warning("Paste a YouTube URL first.")
            else:
                source_path = url_val.strip()
    else:
        uploaded = st.file_uploader(
            "Upload audio",
            type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "webm"],
            label_visibility="collapsed",
            key="file_input",
        )
        tmp_path = None
        if uploaded:
            import tempfile
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
        if st.button("Analyze audio file  →", use_container_width=True, key="btn_aud"):
            if tmp_path:
                source_path = tmp_path
            else:
                st.warning("Upload an audio file first.")


# ── Error ─────────────────────────────────────────────────────────────────────
if st.session_state.error:
    _, ec, _ = st.columns([1, 2.0, 1])
    with ec:
        st.error(f"Processing failed: {st.session_state.error}")
        if st.button("Dismiss", key="dismiss_err"):
            st.session_state.error = None
            st.rerun()


# ── Pipeline ──────────────────────────────────────────────────────────────────
STEPS = [
    "Downloading & extracting audio",
    "Transcribing speech via Whisper",
    "Generating AI summary",
    "Extracting insights & decisions",
    "Building RAG knowledge index",
]

def steps_html(done: int) -> str:
    html = '<div class="ia-steps">'
    for i, label in enumerate(STEPS):
        if i < done:
            cls, prefix = "ia-step-done", "✓ "
        elif i == done:
            cls, prefix = "ia-step-active", ""
        else:
            cls, prefix = "ia-step-idle", ""
        html += (
            f'<div class="ia-step {cls}">'
            f'<div class="ia-step-pip"></div>'
            f'{prefix}{label}</div>'
        )
    html += "</div>"
    return html


def run_pipeline(src: str):
    _, pc, _ = st.columns([1, 2.0, 1])
    with pc:
        st.markdown(
            '<div class="ia-proc-wrap"><div class="ia-proc-card">'
            '<div class="ia-proc-icon">✦</div>'
            '<div class="ia-proc-title">Analyzing content</div>'
            '<div class="ia-proc-sub">// pipeline initiated</div>',
            unsafe_allow_html=True,
        )
        step_ph = st.empty()
        prog = st.progress(0)
        st.markdown("</div></div>", unsafe_allow_html=True)

    try:
        step_ph.markdown(steps_html(0), unsafe_allow_html=True); prog.progress(5)
        chunks = process_input(src)

        step_ph.markdown(steps_html(1), unsafe_allow_html=True); prog.progress(20)
        transcript = transcribe_all(chunks)

        step_ph.markdown(steps_html(2), unsafe_allow_html=True); prog.progress(45)
        title = generate_title(transcript)
        summary_text = summarize(transcript)

        step_ph.markdown(steps_html(3), unsafe_allow_html=True); prog.progress(65)
        action_items = extract_action_items(transcript)
        key_decisions = extract_key_decisions(transcript)
        open_questions = extract_questions(transcript)

        step_ph.markdown(steps_html(4), unsafe_allow_html=True); prog.progress(88)
        rag_chain = build_rag_chain(transcript)

        prog.progress(100)
        time.sleep(0.25)
        step_ph.empty(); prog.empty()

        return {
            "title": title, "transcript": transcript,
            "summary": summary_text, "action_items": action_items,
            "key_decisions": key_decisions, "open_questions": open_questions,
            "rag_chain": rag_chain,
        }
    except Exception as e:
        st.session_state.error = str(e)
        return None


if source_path and source_path != st.session_state.current_source:
    st.session_state.current_source = source_path
    st.session_state.pipeline_result = None
    st.session_state.chat_history = []
    st.session_state.error = None
    result = run_pipeline(source_path)
    if result:
        st.session_state.pipeline_result = result
        st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
result = st.session_state.pipeline_result

if result:
    st.markdown('<div class="ia-divider"></div>', unsafe_allow_html=True)

    title_text = result["title"] or "Untitled Session"
    st.markdown(
        f'<div class="ia-result-header">'
        f'<div class="ia-result-eyebrow">analysis complete</div>'
        f'<div class="ia-result-title">{title_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    word_count = len(result["transcript"].split())
    char_count = len(result["transcript"])
    st.markdown(
        f'<div class="ia-stats">'
        f'<div class="ia-stat">words <span class="ia-stat-val">{word_count:,}</span></div>'
        f'<div class="ia-stat">chars <span class="ia-stat-val">{char_count:,}</span></div>'
        f'<div class="ia-stat">status <span class="ia-stat-val ok">complete</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    def list_html(text: str) -> str:
        lines = [l.strip().lstrip("-*•·").strip() for l in (text or "").strip().split("\n") if l.strip()]
        if not lines:
            return "<span style='color:#1e3a5a;font-style:italic'>Nothing found.</span>"
        items = "".join(f"<li>{l}</li>" for l in lines)
        return f"<ul>{items}</ul>"

    cards = [
        ("✦", "Summary",        "indigo", result["summary"],       "prose"),
        ("✓", "Action Items",   "teal",   result["action_items"],  "list"),
        ("◆", "Key Decisions",  "amber",  result["key_decisions"], "list"),
        ("?", "Open Questions", "violet", result["open_questions"], "list"),
    ]

    cards_html = '<div class="ia-grid">'
    for glyph, label, theme, content, fmt in cards:
        body = (f"<p>{content or '<em style=color:#1e3a5a>Nothing found.</em>'}</p>"
                if fmt == "prose" else list_html(content))
        cards_html += (
            f'<div class="ia-card ia-card-{theme}">'
            f'<div class="ia-card-head"><div class="ia-card-icon">{glyph}</div>'
            f'<span class="ia-card-label">{label}</span></div>'
            f'<div class="ia-card-content">{body}</div>'
            f'</div>'
        )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    _, tc, _ = st.columns([0.08, 2.84, 0.08])
    with tc:
        with st.expander("// view full transcript"):
            st.markdown(
                '<div style="font-family:IBM Plex Mono,monospace;font-size:12px;'
                'line-height:1.9;color:#2d4460;white-space:pre-wrap;padding:8px 4px">'
                + result["transcript"] + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="ia-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="ia-chat-wrap">'
        '<div class="ia-chat-header">'
        '<div class="ia-chat-avatar">✦</div>'
        '<div><div class="ia-chat-title">Ask anything</div>'
        '<div class="ia-chat-desc">// answers grounded in the full transcript</div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    _, chat_col, _ = st.columns([0.2, 2.6, 0.2])
    with chat_col:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
            with st.chat_message("assistant"):
                st.write(entry["answer"])

        if user_q := st.chat_input("Ask about this content..."):
            with st.chat_message("user"):
                st.write(user_q)
            with st.chat_message("assistant"):
                with st.spinner(""):
                    rag = result.get("rag_chain")
                    answer = ask_question(rag, user_q) if rag else "RAG chain unavailable."
                st.write(answer)
            st.session_state.chat_history.append({"question": user_q, "answer": answer})

else:
    st.markdown('<div class="ia-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ia-section-eyebrow">// what insightai does</div>', unsafe_allow_html=True)

    feats = [
        ("🎙", "Transcription",  "rgba(99,102,241,0.5)",  "Full speech-to-text via Whisper AI from YouTube or any audio file."),
        ("✦",  "AI Summary",     "rgba(52,211,153,0.5)",  "Concise summary that captures the core ideas without the noise."),
        ("✓",  "Action Items",   "rgba(20,184,166,0.5)",  "Every task and next step surfaced and flagged clearly."),
        ("◆",  "Key Decisions",  "rgba(245,158,11,0.5)",  "Decisions captured so nothing is lost between meetings."),
        ("?",  "Open Questions", "rgba(139,92,246,0.5)",  "Unresolved topics and follow-ups automatically extracted."),
        ("💬", "RAG Q&A Chat",   "rgba(99,102,241,0.5)",  "Ask anything — the AI reasons over the full transcript."),
    ]

    feat_html = '<div class="ia-features">'
    for glyph, name, accent, desc in feats:
        feat_html += (
            f'<div class="ia-feat" style="--feat-accent:{accent}">'
            f'<span class="ia-feat-glyph">{glyph}</span>'
            f'<div class="ia-feat-name">{name}</div>'
            f'<div class="ia-feat-desc">{desc}</div>'
            f'</div>'
        )
    feat_html += "</div>"
    st.markdown(feat_html, unsafe_allow_html=True)


st.markdown("</div>", unsafe_allow_html=True)