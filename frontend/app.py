import streamlit as st
import time
import html
import base64
import re
from pathlib import Path

# ── 1. CONFIGURAÇÃO DA PÁGINA ───────────────────────────────────────────────
st.set_page_config(page_title="Agente RAG ANEEL", layout="wide", initial_sidebar_state="collapsed")

# ── 2. FUNÇÕES DE INICIALIZAÇÃO (CACHE) ──────────────────────────────────────
@st.cache_resource
def get_cached_rag():
    import sys
    ROOT_PATH = Path(__file__).resolve().parent.parent
    if str(ROOT_PATH) not in sys.path: sys.path.append(str(ROOT_PATH))
    from src.question_rag.fazer_melhorar_perguntas import carregar_banco, criar_llm, criar_retriever, criar_chain_resposta
    return carregar_banco(), criar_llm(), criar_retriever(carregar_banco()), criar_chain_resposta(criar_llm())

# ── 3. CSS DE ALTA PRIORIDADE ───────────────────────────────────────────────
st.markdown("""
    <style>
        header, [data-testid="stHeader"], .stDeployButton { display: none !important; }
        .main .block-container { padding: 0 !important; margin: 0 !important; }
        [data-testid="stStatusWidget"], .stSpinner, [data-testid="stNotification"] { display: none !important; }
        .stApp { background-color: #0A1128 !important; }
    </style>
""", unsafe_allow_html=True)

# ── 4. LÓGICA DA SPLASH SCREEN ──────────────────────────────────────────────
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if not st.session_state.splash_done:
    ROOT = Path(__file__).resolve().parent.parent
    try:
        splash_file = ROOT / "frontend" / "loading-screen (2).html"
        splash_html = splash_file.read_text(encoding="utf-8")
        import streamlit.components.v1 as components
        components.html(splash_html, height=800, scrolling=False)
    except:
        st.markdown("<h1 style='color:white;text-align:center;margin-top:30vh;'>Carregando Sistema...</h1>", unsafe_allow_html=True)

    with st.empty(): get_cached_rag()
    time.sleep(6.0) 
    st.session_state.splash_done = True
    st.rerun()

# ── 5. IMPORTAÇÕES E CSS PÓS-CARREGAMENTO ───────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
from src.question_rag.fazer_melhorar_perguntas import formatar_contexto, recuperar_documentos, refinar_pergunta

st.markdown("""
    <style>
    :root {
        --bg-panel: rgba(18, 30, 56, 0.85);
        --accent-color: #00D2FF;
        --border-color: #00D2FF;
    }
    #neural-bg { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1; overflow: hidden; }
    .stApp { background-color: transparent !important; }
    .block-container { padding: 1rem 3rem !important; }
    
    .chat-container {
        display: flex; flex-direction: column-reverse; height: 65vh;
        overflow-y: auto !important; padding: 20px; border: 2px solid var(--border-color) !important;
        border-radius: 15px; background-color: var(--bg-panel);
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.2); backdrop-filter: blur(8px);
    }
    
    .bubble { padding: 12px 18px; border-radius: 18px; margin-bottom: 12px; max-width: 78%; font-family: 'Inter', sans-serif; font-size: 15px; line-height: 1.5; display: inline-block; white-space: pre-wrap; }
    .user-wrapper { display: flex; justify-content: flex-end; width: 100%; }
    .user-bubble { background-color: #004B87; color: white; border-bottom-right-radius: 2px; }
    .ai-wrapper { display: flex; justify-content: flex-start; width: 100%; }
    .ai-bubble { background-color: #1E293B; color: #F8FAFC; border-bottom-left-radius: 2px; border: 1px solid var(--border-color) !important; }
    
    .stChatInputContainer { background-color: transparent !important; padding-bottom: 20px; }
    .stChatInput { border: 2px solid var(--border-color) !important; border-radius: 12px !important; background-color: #FFFFFF !important; }
    .stChatInput textarea { color: #000000 !important; font-size: 16px !important; caret-color: #000000 !important; }

    .stButton>button { background-color: #004B87 !important; color: white !important; border: 1px solid #00D2FF !important; border-radius: 8px !important; }
    
    .header-container { display: flex; align-items: center; gap: 25px; margin-bottom: 30px; }
    .ceinha-avatar {
        width: 160px; height: 160px; border-radius: 50%;
        border: 4px solid var(--accent-color); overflow: hidden;
        background-color: #000; box-shadow: 0 0 25px rgba(0, 212, 255, 0.6);
        position: relative; flex-shrink: 0;
    }
    .ceinha-iframe { position: absolute; top: 50%; left: 50%; width: 800px; height: 800px; border: none; transform: translate(-50%, -50%) scale(0.28); }

    /* FONTES - VISIBILIDADE MÁXIMA FORÇADA */
    [data-testid="stSidebar"] *, [data-testid="column"]:nth-child(2) * { color: #FFFFFF !important; }
    [data-testid="stExpander"] { background-color: rgba(18, 30, 56, 0.98) !important; border: 1.5px solid #00D2FF !important; }
    [data-testid="stExpander"] p, [data-testid="stExpander"] div, [data-testid="stExpander"] span { color: #FFFFFF !important; font-size: 15px !important; opacity: 1 !important; }
    [data-testid="stExpander"] .stCaption p { color: #00D2FF !important; font-weight: bold !important; }

    header, footer {visibility: hidden !important;}
    </style>
""", unsafe_allow_html=True)

# ── 6. INJEÇÃO DO FUNDO ANIMADO E AVATAR ────────────────────────────────────
try:
    bg_path = ROOT / "frontend" / "neural-background.html"
    b64_bg = base64.b64encode(bg_path.read_text(encoding="utf-8").encode("utf-8")).decode()
    st.markdown(f'<div id="neural-bg"><iframe src="data:text/html;base64,{b64_bg}" style="width:100%; height:100%; border:none;" scrolling="no"></iframe></div>', unsafe_allow_html=True)
except: pass

try:
    ceinha_path = ROOT / "frontend" / "CEINHA.html"
    ceinha_clean = ceinha_path.read_text(encoding="utf-8").replace('<div class="brand">CEINHA</div>', '').replace('background: radial-gradient', 'background: #000')
    b64_ceinha = base64.b64encode(ceinha_clean.encode("utf-8")).decode()
    ceinha_src = f"data:text/html;base64,{b64_ceinha}"
except: ceinha_src = ""

st.markdown(f'<div class="header-container"><div class="ceinha-avatar"><iframe src="{ceinha_src}" class="ceinha-iframe" scrolling="no"></iframe></div><h1 style="color: #00D2FF; margin: 0; font-size: 3.5rem; font-weight: 800; text-shadow: 0 0 15px rgba(0,210,255,0.5);">Chat RAG ANEEL</h1></div>', unsafe_allow_html=True)

# ── 7. LÓGICA DO CHAT ───────────────────────────────────────────────────────
banco, llm, retriever, chain = get_cached_rag()

if "messages" not in st.session_state: st.session_state.messages = []
if "last_sources" not in st.session_state: st.session_state.last_sources = []

col_chat, col_side = st.columns([3, 1], gap="medium")

with col_chat:
    chat_placeholder = st.empty()

    if prompt := st.chat_input("Pergunte à Ceinha..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

    with chat_placeholder:
        chat_html = '<div class="chat-container" id="chat-container">'
        for msg in reversed(st.session_state.messages):
            txt = html.escape(msg["content"])
            if msg["role"] == "user":
                chat_html += f'<div class="user-wrapper"><div class="bubble user-bubble">{txt}</div></div>'
            else:
                chat_html += f'<div class="ai-wrapper"><div class="bubble ai-bubble">{txt}</div></div>'
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    if prompt:
        with st.spinner(""):
            try:
                p_limpa = refinar_pergunta(llm, prompt)
                docs = recuperar_documentos(retriever, p_limpa)
                st.session_state.last_sources = docs
                
                if docs:
                    resp = chain.invoke({"contexto": formatar_contexto(docs), "pergunta": p_limpa})
                    resp = resp.replace('<div', '').replace('</div>', '').replace('<span', '').replace('</span>', '')
                else:
                    resp = "Não localizei informações suficientes nos documentos para responder a essa pergunta."
                
                st.session_state.messages.append({"role": "assistant", "content": resp})
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate_limit" in error_msg.lower():
                    st.error("Limite de requisições atingido (Groq 429). Aguarde alguns segundos e tente novamente.")
                    st.session_state.messages.append({"role": "assistant", "content": "⚠️ Erro de Limite (Rate Limit): O servidor do Groq está temporariamente sobrecarregado. Por favor, aguarde cerca de 10 segundos antes de perguntar novamente."})
                else:
                    st.error(f"Erro inesperado: {error_msg}")
                    st.session_state.messages.append({"role": "assistant", "content": "⚠️ Desculpe, ocorreu um erro ao processar sua pergunta. Por favor, tente novamente."})
            st.rerun()



with col_side:
    st.markdown('<h3 style="color: #00D2FF;">📄 Fontes</h3>', unsafe_allow_html=True)
    if st.session_state.last_sources:
        for i, doc in enumerate(st.session_state.last_sources):
            with st.expander(f"Trecho {i+1}"):
                st.caption(doc.metadata.get("file_name", "Arquivo"))
                st.write(doc.page_content)
        if st.button("Limpar Histórico"):
            st.session_state.messages = []; st.session_state.last_sources = []; st.rerun()
    else:
        st.markdown('<p style="color: #94A3B8;">Fontes aparecerão aqui.</p>', unsafe_allow_html=True)
