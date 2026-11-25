#  SovTeh AI Plot & Style Agent: Гибридный RAG (RuBERT + FAISS), Локальный ML, Универсальный LLM-шлюз, MLOps и Агентные циклы

Автономный ИИ-агент на базе гибридной архитектуры, включающей локальное семантическое ядро **RuBERT + FAISS**, кастомную **RAG-систему** (Retrieval-Augmented Generation), локальный ML-фильтр (**`HashingVectorizer` + `SGDClassifier`** и нейросетевые классификаторы **`ruBERT-base + LoRA` / `char TF-IDF + PyTorch MLP`**), MLOps-конвейер (**MLflow**) и бэкенд на **FastAPI**.

Агент выполняет интеллектуальный контекстный анализ сюжетов из подготовленного датасета (`data/plots_dataset.csv`) и генерирует точные текстовые ответы через API-ключ **любой модели по API** (**Google AI Studio / Gemini**, **Groq Cloud**, **OpenRouter**, **OpenAI**, **Sber GigaChat** или любой OpenAI-совместимый / локальный эндпоинт `LLM_BASE_URL`), обходя любые сетевые прокси-блокировки Windows.

---

##  Ключевые особенности и архитектура

1. **Двухэтапный конвейер (Пайплайн) обработки запроса**:
   - **Локальный ML-фильтр (`src/rag.py`, `src/classifier.py`)**: Предварительно обученные модели `HashingVectorizer` + `SGDClassifier` моментально категоризируют жанровую направленность вопроса пользователя (7 макро-жанров), а локальные классификаторы **`ruBERT-base + LoRA`** и **`PyTorch Baseline (char TF-IDF + MLP)`** определяют ближайший авторский стиль из **64 русских классиков**.
   - **Семантический поиск (RuBERT + FAISS)**: Локальное семантическое ядро (`ai-forever/ruBert-base` + гибридный плотный индекс) и векторный индекс **FAISS** (`IndexFlatIP` с прозрачным PyTorch/NumPy fallback) осуществляют высокоточный поиск по векторам описаний сюжетов в оперативной памяти (с поддержкой аппаратного ускорения **NVIDIA CUDA GPU** и CPU).
2. **Динамический шлюз LLM API (любая модель по API — `src/llm_client.py`, `src/config.py`)**:
   - Поддержка произвольного провайдера и модели: **Google AI Studio (`AQ.`/`AIza`)**, **Groq (`gsk_`)**, **OpenRouter (`sk-or-`)**, **OpenAI (`sk-`)**, **Sber GigaChat (`GIGACHAT_CREDENTIALS`)** или пользовательского эндпоинта (`LLM_BASE_URL` + `LLM_MODEL`).
   - Сервер при старте автоматически опрашивает эндпоинт доступных моделей провайдера (`GET /models`), отфильтровывает эмбеддинговые/аудио модели, выявляет доступные текстовые модели вашего тарифа и на лету подстраивает вызовы, полностью защищая систему от `404` ошибок при обновлении линеек моделей.
3. **Сетевой иммунитет (`src/preprocessing.py`, `src/app.py`, `src/llm_client.py`, `ui.py`)**:
   - В код бэкенда и клиентов внедрена жёсткая изоляция процесса Python (`enforce_network_immunity()` + `trust_env=False`) от системных прокси-серверов, VPN-клиентов и перехватчиков сетевого стека Windows (LSP/Tun/Tap), что решает проблему блокировок `403 Forbidden` и обрывов сокетов.
4. **Кросс-платформенное кодирование (`UTF-8` / `CP1251`)**:
   - Автоматическое чтение текстовых файлов и CSV-датасетов в `UTF-8` (с BOM и без) и `CP1251` (`read_text_cross_platform`), переконфигурация потоков консоли (`configure_cross_platform_stdio`) и принудительный заголовок `UTF8JSONResponse` (`application/json; charset=utf-8`) для бесшовной интеграции как в веб-интерфейс Streamlit, так и в классическую консоль Windows PowerShell.

---

##  Архитектура по слоям (Что, где и почему)

### 1. Слой предобработки, ML/DL-классификации и подготовки весов (`src/preprocessing.py`, `src/classifier.py`, `notebooks/`)
* **Классификация стиля (64 автора):**
  * **`lora` *(по умолчанию)* (`LoraClassifier`):** `ai-forever/ruBert-base` с адаптером PEFT LoRA (`r=16, alpha=32`, **1.35%** обучаемых параметров, вес адаптера **9.30 МБ**). **Accuracy = 99.15%**, **Macro-F1 = 98.53%** по целым документам (`91.72% / 85.07%` на фрагментах ~1000 симв.).
  * **`baseline` (`BaselineClassifier`):** посимвольный `TfidfVectorizer` (`char`, n-граммы `(2, 4)`, 50 000 признаков, `sublinear_tf=True`) + PyTorch `TextClassifier` (`Linear(50000->128) -> ReLU -> Linear(128->64)`). **Accuracy = 99.29%**, **Macro-F1 = 96.85%** по документам (`94.35% / 88.67%` на чанках) при задержке **`1.99 мс`** на CPU и **`1.34 мс`** на GPU.
* **Предобработка и изоляция сети (`src/preprocessing.py`):** единая очистка и нарезка `chunk_text`, `enforce_network_immunity()`, `configure_cross_platform_stdio()` и `read_text_cross_platform()` (`UTF-8` / `CP1251`).

### 2. Семантическое ядро RAG и REST API бэкенд (`src/rag.py`, `src/app.py`, `src/llm_client.py`, `src/config.py`)
* **Подготовленный датасет (`data/plots_dataset.csv`):** 132 подробных описания сюжетов по всем **64 авторам** из `models/lora_classifier/labels.json`, размеченных по 7 макро-жанрам и рейтингам.
* **Двухэтапный RAG-поиск (`rag_search`):**
  1. `HashingVectorizer` + `SGDClassifier` (`predict_genre`) и `StyleClassifier` определяют жанр и ближайший авторский стиль.
  2. `FAISSVectorIndex` (RuBERT/плотные L2-нормализованные векторы + `faiss.IndexFlatIP` / PyTorch GPU/CPU) и лексический скоринг по корням находят наиболее близкие сюжеты. При отсутствии точных совпадений система честно предупреждает пользователя и предлагает топ-произведения по рейтингу.
* **Динамический шлюз (`LLMClient`):** опрос `GET /models`, автоматический fallback при `404`/`429`/`5xx`, защита от prompt-инъекций (`build_rag_system_prompt`).

### 3. MLOps-конвейер, реестр моделей и CI (`src/train_mlflow.py`, `.github/workflows/ci.yml`)
* Локальное хранилище **MLflow (`sqlite:///mlflow.db`)**, эксперимент **`doc-classifier`**, сетки гиперпараметров (`--quick` и полная сетка из 9 запусков), артефакты (`config/run_config.json`, `charts/accuracy_chart.png`) и регистрация лучшей модели **`literary-style-classifier`** в стадии **`Staging`** (`@staging`).
* GitHub Actions CI (`.github/workflows/ci.yml`): автоматический запуск `ruff` и автономного `pytest` на каждый `push`/`pull_request` в `main`.

### 4. Автономные ИИ-агенты и Мультиагентный UI (`src/agents.py`, `agents_comparison.py`, `ui.py`)
* **4 инструмента с Pydantic-валидацией (`extra="forbid"`) и безопасной обёрткой `run_tool()`:**
  1. `classify_style` — локальная атрибуция стиля (64 автора, топ-3).
  2. `search_plots` — семантический поиск сюжетов в `data/plots_dataset.csv`.
  3. `author_metrics` — поиск тестовых метрик (`precision`, `recall`, `f1_score`, `support`) в `reports/`.
  4. `calc` — безопасный AST-калькулятор (`+`, `-`, `*`, `/`, скобки).
* **Сравнение 3 агентных циклов (`agents_comparison.py`):** `run_native_agent` (Native Tool Calling), `run_json_agent` (ручной JSON-цикл), `run_framework_agent` (LangGraph ReAct + MLflow Tracing).
* **Веб-интерфейс Streamlit (`ui.py`):** 3 режима работы — *RAG-Эксперт по сюжетам*, *Агент-Аналитик (статистика и рейтинги)* и *Агент-Критик (сравнение кандидатов)*.

---

##  Структура репозитория

```text
SovTeh/
├── .github/workflows/ci.yml         # CI в GitHub Actions: линтер ruff + автономный pytest
├── data/
│   └── plots_dataset.csv            # Подготовленный датасет сюжетов (132 записи, все 64 автора)
├── src/
│   ├── __init__.py                  # Маркер пакета src
│   ├── app.py                       # FastAPI-сервер (/, /health, /classify, /ask, /agent, lifespan)
│   ├── rag.py                       # Семантическое ядро RAG (Hashing+SGD фильтр, RuBERT+FAISS, Аналитик/Критик)
│   ├── classifier.py                # Классификаторы BaselineClassifier (MLP) и LoraClassifier (ruBERT+LoRA)
│   ├── preprocessing.py             # Очистка, чанкинг, UTF-8/CP1251 кодирование, сетевой иммунитет
│   ├── config.py                    # Настройки pydantic-settings и автоопределение любых LLM-провайдеров
│   ├── schemas.py                   # Pydantic-контракты для /health, /classify, /ask, /agent
│   ├── llm_client.py                # Динамический шлюз LLM API (GET /models, ретраи, fallback, trust_env=False)
│   ├── train_mlflow.py              # MLOps-пайплайн: эксперименты MLflow, графики, артефакты и Staging
│   └── agents.py                    # 4 инструмента, run_tool, Native / JSON / LangGraph циклы + MLflow Tracing
├── ui.py                            # Мультиагентный веб-интерфейс на Streamlit (порт 8501)
├── test_request.py                  # Изолированный тестовый клиент с обходом прокси Windows
├── agents_comparison.py             # Сравнение 3 архитектур ИИ-агентов (--mock и --trace режимы)
├── models/                          # Обученные веса и векторизатор (char TF-IDF, PyTorch MLP, ruBERT+LoRA)
├── notebooks/                       # 4 исследовательских ноутбука (01_eda..04_lora_finetuning)
├── reports/                         # Метрики (metrics_*.json, lora_per_class_report.csv) и графики (figures/)
├── tests/                           # 8 модулей автотестов (44 сценария)
├── docs/                            # USER_GUIDE.md, SWAGGER_DOCS_GUIDE.md, CODEBASE_ARCHITECTURE.md, TECHNICAL_DOCUMENTATION.md
├── Dockerfile                       # Сборка изолированного CPU-контейнера с RuBERT и data/plots_dataset.csv
├── requirements.txt                 # Зависимости проекта
├── .env.example                     # Шаблон настроек для любой модели по API
├── .dockerignore
└── .gitignore
```

---

##  Инструкция по запуску и демонстрации

### 1. Установка зависимостей и настройка ключа любой LLM
```bash
# Создание и активация виртуального окружения (Python 3.11 / 3.12)
python -m venv .venv
source .venv/bin/activate        # В Windows PowerShell: .venv\Scripts\Activate.ps1

# Лёгкая установка PyTorch (CPU) или CUDA 12.8 для GPU + зависимости проекта
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Создайте в корне файл `.env` на основе `.env.example` и укажите ключ **любого API-провайдера** (Google AI Studio, Groq, OpenRouter, OpenAI, GigaChat или локальный Ollama/vLLM через `LLM_BASE_URL` + `LLM_MODEL`):
```env
LLM_API_KEY="Ваш_API_Ключ_Любой_LLM"
LLM_PROVIDER=auto
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o-mini
LLM_AUTO_DISCOVER_MODELS=true
BYPASS_SYSTEM_PROXY=true
CLASSIFIER_BACKEND=lora
DEVICE=auto
```

### 2. Запуск FastAPI-сервера и веб-интерфейса Streamlit
**Терминал 1 — запуск REST API сервера:**
```bash
uvicorn src.app:app --host localhost --port 8000 --reload
```
Документация Swagger UI: **`http://localhost:8000/docs`**.

**Терминал 2 — запуск веб-интерфейса Streamlit (опционально, требует `pip install streamlit`) или проверка клиентом:**
```bash
python -m streamlit run ui.py
# Либо быстрый тест из консоли (с изоляцией прокси Windows):
python test_request.py
```

**Проверка напрямую из Windows PowerShell:**
```powershell
$Body = '{"question": "Найди сюжет про космическую экспедицию к Марсу и расскажи, что там происходит", "k": 3}'
$Response = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/ask" -Body ([System.Text.Encoding]::UTF8.GetBytes($Body)) -ContentType "application/json; charset=utf-8"
$Response.answer
```

**Проверка основных HTTP-эндпоинтов через `curl`:**
```bash
# 1. Готовность сервиса
curl http://localhost:8000/health

# 2. Локальная классификация стиля (офлайн, без LLM_API_KEY)
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"question": "В саду за старой усадьбой тихо осыпались липы; доктор пил остывший чай на террасе.", "k": 3}'

# 3. Контекстный RAG-анализ сюжетов + определение жанра и стиля
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Найди сюжет про космическую экспедицию к Марсу", "k": 3, "temperature": 0.3}'

# 4. Многошаговый ИИ-агент (approach: "native" | "json" | "framework", либо UI-режим mode: "Аналитик" | "Критик")
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "Узнай F1-score Чехова А и Достоевского Ф и вычисли разницу", "approach": "native"}'
```

### 3. Демонстрация MLOps и реестра моделей (MLflow)
```bash
python -m src.train_mlflow --quick
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Веб-интерфейс будет доступен по адресу **`http://127.0.0.1:5000`** (эксперимент **`doc-classifier`**, модель **`literary-style-classifier`** в стадии **`Staging`**).

### 4. Демонстрация 3 подходов ИИ-агентов
```bash
# Автономный демо-прогон с замоканной LLM и реальными локальными инструментами:
python agents_comparison.py --mock --trace

# Живой прогон с реальным LLM API из .env + запись MLflow Traces:
python agents_comparison.py --trace
```

### 5. Проверка качества (линтер, автотесты и Docker)
```bash
# Линтер Ruff и автономные тесты (соответствует GitHub Actions CI)
ruff check src tests agents_comparison.py ui.py test_request.py --line-length 105
CLASSIFIER_BACKEND=baseline pytest tests -v --cov=src --cov-report=term-missing -k "not test_live_e2e and not test_baseline_and_lora"

# Сборка и запуск в Docker
docker build -t sovteh-rag-agent .
docker run --rm -p 8000:8000 -e LLM_API_KEY="Ваш_API_Ключ_LLM" sovteh-rag-agent
```

Дополнительные материалы в репозитории:
* [REPORT.md](REPORT.md) — аналитический отчёт по экспериментам, метрикам, RAG и покрытию тестами.
* [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — простое объяснение проекта и подробное руководство пользователя.
* [docs/SWAGGER_DOCS_GUIDE.md](docs/SWAGGER_DOCS_GUIDE.md) — полная пошаговая инструкция по каждому действию в Swagger UI (`http://localhost:8000/docs`).
* [docs/CODEBASE_ARCHITECTURE.md](docs/CODEBASE_ARCHITECTURE.md) — карта кода с разбором каждого модуля.
* [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) — инженерная спецификация API, RAG, шлюза LLM, MLOps и агентов.
