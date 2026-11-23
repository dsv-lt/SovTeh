"""Мультиагентный веб-интерфейс (Streamlit UI) для гибридной RAG-системы SovTeh."""

from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="SovTeh AI Эксперт по сюжетам", page_icon="📚", layout="wide")

with st.sidebar:
    st.header("🤖 Выбор режима агента")
    agent_mode = st.radio(
        "Выберите активного агента:",
        [
            "🎬 1. RAG-Эксперт (Поиск по сюжетам и стилю)",
            "📊 2. Агент-Аналитик (Цифры и статистика базы)",
            "⚖️ 3. Агент-Критик (Сравнение и выбор сюжета)",
        ],
    )

    st.markdown("---")
    if "RAG" in agent_mode:
        k_docs = st.slider("Количество сюжетов в выборке (k)", 1, 10, 3)
        st.caption(
            "Двухэтапный пайплайн: локальный ML-фильтр жанра/стиля + семантический "
            "поиск RuBERT/FAISS по датасету сюжетов + точный ответ выбранной LLM."
        )
    elif "Аналитик" in agent_mode:
        st.caption(
            "Сканирует локальный датасет сюжетов для подсчёта средних рейтингов, "
            "сравнения жанров и поиска экстремумов."
        )
    else:
        st.caption(
            "Подбирает кандидатов из датасета и сравнивает их по сюжету, атмосфере, "
            "авторскому стилю и рейтингу."
        )

    st.markdown("---")
    api_url = st.text_input("URL бэкенда FastAPI", value="http://localhost:8000")
    if st.button("🗑️ Очистить историю чата"):
        st.session_state.messages = []
        st.rerun()

st.title(agent_mode)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.caption(msg["meta"])
        if msg.get("sources"):
            with st.expander("📚 Источники / Шаги работы агента"):
                for src in msg["sources"]:
                    st.text(src)

placeholder = (
    "Например: найди произведение про космическую экспедицию к Марсу и расскажи сюжет..."
    if "RAG" in agent_mode
    else "Например: сравни средний рейтинг фантастики и драмы в базе..."
)

if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"), st.spinner("Агент анализирует датасет..."):
        try:
            session = requests.Session()
            session.trust_env = False  # Изоляция от системных прокси Windows

            base = api_url.rstrip("/")
            if "RAG" in agent_mode:
                resp = session.post(
                    f"{base}/ask",
                    json={"question": prompt, "k": k_docs},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or data.get("generated_text") or "Нет ответа"
                sources = data.get("sources", [])
                meta = (
                    f"Жанр (ML-фильтр): {data.get('predicted_genre', 'Н/Д')} | "
                    f"Ближайший стиль: {data.get('predicted_author', 'Н/Д')} "
                    f"({data.get('confidence', 0):.2%})"
                )
            else:
                resp = session.post(
                    f"{base}/agent",
                    json={"question": prompt, "mode": agent_mode},
                    timeout=90,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer", "Нет ответа")
                sources = [
                    str(step) if isinstance(step, str) else f"{step.get('tool')}: {step.get('result')}"
                    for step in data.get("steps", [])
                ]
                meta = f"Режим агента: {data.get('approach', 'agent')}"

            st.markdown(answer)
            if meta:
                st.caption(meta)
            if sources:
                with st.expander("📚 Источники / Шаги работы агента"):
                    for src in sources:
                        st.text(src)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "meta": meta,
                    "sources": sources,
                }
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ошибка подключения к API: {exc}")
