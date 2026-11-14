# 📚 AI Literary Style Assistant: Классификация стиля (64 автора), Стилизация, MLOps и Агентные циклы

Двухстадийный ИИ-микросервис и исследовательский конвейер для атрибуции литературного стиля по текстам **64 русских классиков** и стилизованной генерации текста.

Проект объединяет **нейросетевое машинное обучение (`char` TF-IDF + PyTorch MLP и `ruBERT-base + LoRA`)**, **асинхронный REST API на FastAPI** с внешними LLM (**Google AI Studio**, **Groq Cloud**, **OpenRouter**), **MLOps-конвейер (MLflow Tracking, Artifacts и Model Registry `Staging`)**, **CI в GitHub Actions (`ruff` + `pytest`)** и **сравнение трёх архитектур ИИ-агентов** с вызовом инструментов и MLflow Tracing.

---

## 🧠 Архитектура и логика работы (Что, где и почему)

Система разделена на **4 независимых слоя**, каждый из которых решает конкретную инженерную задачу:

### 1. Слой предобработки, ML/DL-классификации и подготовки весов (`src/preprocessing.py`, `src/classifier.py`, `notebooks/`)
* **Зачем это нужно:** Определять авторский почерк локально на видеокарте или CPU за миллисекунды — без потери редких писателей из **64 классов** корпуса [`ViliamVolosV/russian_classic_literature`](https://huggingface.co/datasets/ViliamVolosV/russian_classic_literature) и без расхода токенов облачной LLM на классификацию.
* **Как реализовано:**
  * Реализована единая предобработка (`src/preprocessing.py`) для обучения и продакшена (*Zero Train-Serving Skew*): двухуровневое извлечение авторов (`AUTHOR_RE_PRIMARY` + `AUTHOR_RE_FALLBACK`, 100% покрытие 4 шаблонов), очистка битых экранирований и невидимых/управляющих символов (`clean_text`), нормализация и нарезка по границам слов на фрагменты ~1000 символов (`chunk_text`) с усреднением `softmax`-вероятностей по чанкам.
  * Обучены два взаимозаменяемых бэкенда (`CLASSIFIER_BACKEND`):
    * **`lora` *(по умолчанию)* (`LoraClassifier`):** `ai-forever/ruBert-base` с адаптером PEFT LoRA (`r=16, alpha=32`, обучалось **1.35%** весов — 2.43M из 180.8M параметров), слиянием весов `.merge_and_unload()` на старте и `bfloat16` на CUDA.
    * **`baseline` (`BaselineClassifier`):** посимвольный `TfidfVectorizer` (`char`, n-граммы `(2, 4)`, 50 000 признаков, `sublinear_tf=True`, без лемматизации, чтобы сохранять авторскую пунктуацию и частицы вроде *«-с»*) + PyTorch `TextClassifier` (`Linear(50000->128) -> ReLU -> Linear(128->64)`, сырые логиты без `Softmax`).
* **Результат:**
  * `ruBERT-base + LoRA`: **Accuracy = 99.15%**, **Macro-F1 = 98.53%** по целым документам (`91.72% / 85.07%` на фрагментах ~1000 симв.), вес адаптера **9.30 МБ**.
  * `PyTorch Baseline`: **Accuracy = 99.29%**, **Macro-F1 = 96.85%** по целым документам (**`94.35% / 88.67%`** на фрагментах ~1000 симв.) при задержке **`1.99 мс`** на CPU и **`1.34 мс`** на GPU.
  * Артефакты сохранены в `models/` (`tfidf_vectorizer.pkl`, `label_encoder.pkl`, `baseline_pytorch.pt`, `lora_classifier/`).

### 2. Слой стилизованной генерации и REST API бэкенд (`src/app.py`, `src/llm_client.py`, `src/config.py`, `src/schemas.py`)
* **Зачем это нужно:** Предоставить единый HTTP-интерфейс (`GET /health`, `POST /classify`, `POST /ask`, `POST /agent`), исключив повторную загрузку весов на запросах и сбои при недоступности облачного API.
* **Как реализовано:**
  * **`lifespan` в `src/app.py`:** загружает векторизатор, классификатор и пул соединений `httpx.AsyncClient` один раз при старте сервера; тяжёлый PyTorch-инференс вынесен в `asyncio.to_thread()` под потоковой блокировкой `threading.Lock()`.
  * **Универсальный клиент (`src/llm_client.py`):** метод `LLMClient.chat(messages, tools, temperature)` поддерживает как обычную генерацию (`generate()`), так и нативный вызов инструментов (`ToolCall`), а `Settings` (`src/config.py`) хранит ключ в `SecretStr` и автоматически определяет провайдера по префиксу (`AQ.`/`AIza` → **Google AI Studio**, `gsk_` → **Groq Cloud**, `sk-or-` → **OpenRouter**).
  * **Защита от сбоев и атак:** системный промпт `build_system_prompt()` изолирует текст в тройных кавычках и блокирует prompt-инъекции; пустые/невидимые строки отклоняются с `400 {"detail": "Пустой вопрос"}`; на `429/5xx` и тайм-аутах выполняются экспоненциальный backoff и переключение на резервные модели (`fallback_models`).
* **Результат:** `POST /classify` работает полностью офлайн без ключа, `POST /ask` возвращает автора + уверенность + стилизованный текст, `POST /agent` запускает любой из 3 агентных циклов, а ответы принудительно кодируются через `UTF8JSONResponse`.

### 3. MLOps-конвейер, реестр моделей и CI (`src/train_mlflow.py`, `.github/workflows/ci.yml`)
* **Зачем это нужно:** Автоматизировать подбор гиперпараметров, логирование метрик и артефактов (графиков, конфигов, моделей), версионирование лучшей модели и проверку качества кода при каждом изменении репозитория.
* **Как реализовано:**
  * Настроено локальное хранилище **MLflow (`sqlite:///mlflow.db`)** и эксперимент **`doc-classifier`**.
  * Прогоняется серия экспериментов (`train_once`: по умолчанию сетка из **9 запусков** `lr ∈ {0.0001, 0.001, 0.01} × epochs ∈ {5, 10, 20}`, быстрый режим `--quick` из **4 запусков** и автоматический воспроизводимый `synthetic_fallback` при отсутствии `data/processed/chunks_train.parquet`).
  * Для каждого запуска логируются параметры, метрики (`train_loss` по эпохам, `test_accuracy`, `test_f1_macro`, `train_time_sec`), теги (`assignment=mlops-task-1`), артефакты (`config/run_config.json`, сравнительная диаграмма `charts/accuracy_chart.png`) и PyTorch-модель через `mlflow.pytorch.log_model` с `infer_signature` и `input_example`.
  * В GitHub Actions (`.github/workflows/ci.yml`) на каждый `push` и `pull_request` в `main` автоматически запускаются `ruff` и автономный `pytest`.
* **Результат:** Лучшая модель автоматически регистрируется в **MLflow Model Registry** под именем **`literary-style-classifier`** и переводится в стадию **`Staging`** (алиас `@staging`, тег `stage=Staging` и stage-переход).

### 4. Автономные агентные циклы и Tool Calling (`src/agents.py`, `agents_comparison.py`)
* **Зачем это нужно:** Научить систему решать **составные многошаговые задачи** (например: *«Определи автора отрывка, узнай F1-score для найденного автора и Достоевского Ф, а затем вычисли точную разницу между ними через калькулятор»*), комбинируя локальную нейросеть, аналитику отчётов и точные вычисления.
* **Реализованы 3 инструмента** с docstring-контрактами, Pydantic-валидацией (`extra="forbid"`) и защищённой обёрткой `run_tool(name, args)`:
  1. `classify_style` — локальная атрибуция стиля отрывка через `StyleClassifier` (`64` автора, топ-3).
  2. `author_metrics` — поиск тестовых метрик (`precision`, `recall`, `f1_score`, `support`) по автору или сводки `'overall'` в `reports/`.
  3. `calc` — безопасный AST-калькулятор для арифметических выражений (`+`, `-`, `*`, `/`, скобки) с обработкой деления на ноль.
* **Сравнение 3 архитектурных подходов** (с поддержкой консольных логов, автономного `--mock` режима и **MLflow Tracing** в эксперименте `doc-classifier-agents`):
  1. **Native Tool Calling (`run_native_agent`):** передача автогенерированных схем `TOOL_SCHEMAS` в `llm_client.chat(..., tools=...)`, ручная обработка `tool_calls` и возврат `role="tool"` по `tool_call_id`.
  2. **Ручной JSON-цикл (`run_json_agent`):** модель на каждом шаге возвращает строгий JSON `{"action": "...", "input": "..."}`, парсер `extract_json` (очистка markdown + `JSONDecoder.raw_decode`) извлекает команду, а ошибки парсинга/инструментов возвращаются в историю для самоисправления.
  3. **Фреймворк LangGraph (`run_framework_agent`):** автоматическая ReAct-оркестрация через `langgraph.prebuilt.create_react_agent` с `ChatOpenAI`, `@tool`-обёртками, защитой `GraphRecursionError` и автологированием `mlflow.langchain.autolog()`.

---

## 📂 Структура репозитория

```text
SovTeh/
├── .github/
│   └── workflows/
│       └── ci.yml                   # CI в GitHub Actions: линтер ruff + автономный pytest
├── src/
│   ├── __init__.py                  # Маркер пакета src
│   ├── app.py                       # FastAPI-сервер (/, /health, /classify, /ask, /agent, lifespan)
│   ├── classifier.py                # Классификаторы BaselineClassifier (MLP) и LoraClassifier (ruBERT+LoRA)
│   ├── preprocessing.py             # Единая очистка, парсинг 4 шаблонов авторов и нарезка chunk_text
│   ├── config.py                    # Настройки pydantic-settings и автоопределение провайдеров LLM
│   ├── schemas.py                   # Pydantic-контракты для API-эндпоинтов и шагов агента
│   ├── llm_client.py                # Асинхронный клиент OpenAI-совместимых LLM (chat, tool_calls, generate)
│   ├── train_mlflow.py              # MLOps-пайплайн: эксперименты MLflow, графики, артефакты и Staging
│   └── agents.py                    # 3 инструмента, run_tool, Native / JSON / LangGraph циклы + MLflow Tracing
├── agents_comparison.py             # Сравнение 3 архитектур ИИ-агентов (--mock и --trace режимы)
├── models/                          # Обученные веса и векторизатор (char TF-IDF, PyTorch MLP, ruBERT+LoRA)
├── notebooks/                       # 4 исследовательских ноутбука (01_eda..04_lora_finetuning)
├── reports/                         # Метрики (metrics_*.json, lora_per_class_report.csv) и графики (figures/)
├── tests/                           # 7 модулей автотестов (38 сценариев + живой E2E-тест)
├── docs/                            # Подробные руководства: USER_GUIDE, CODEBASE_ARCHITECTURE, TECHNICAL_DOCUMENTATION
├── Dockerfile                       # Сборка изолированного CPU-контейнера с предзагруженным ruBERT-base
├── requirements.txt                 # Зависимости проекта (PyTorch, Transformers, FastAPI, MLflow, LangGraph, Ruff, Pytest)
├── .env.example                     # Безопасный шаблон настроек без секретов
├── .dockerignore                    # Исключения для сборки Docker-образа
└── .gitignore                       # Исключение data/, .venv/, mlflow.db, mlruns/, временных графиков и .env
```

---

## 🚀 Инструкция по запуску и демонстрации

### 1. Установка зависимостей и настройка ключа
```bash
# Создание и активация виртуального окружения (Python 3.11 / 3.12)
python -m venv .venv
source .venv/bin/activate        # В Windows PowerShell: .venv\Scripts\Activate.ps1

# Лёгкая установка PyTorch (CPU) или CUDA 12.8 для GPU + зависимости проекта
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
Создайте в корне файл `.env` на основе `.env.example` и добавьте ключ внешнего LLM API (**Google AI Studio**, **Groq** или **OpenRouter** определяется автоматически):
```env
LLM_API_KEY="Ваш_API_Ключ_LLM"
LLM_PROVIDER=auto
CLASSIFIER_BACKEND=lora
DEVICE=auto
```

### 2. Запуск основного REST API сервера и проверка эндпоинтов
Запустите FastAPI-сервер:
```bash
uvicorn src.app:app --host localhost --port 8000 --reload
```
Интерактивная документация Swagger UI откроется по адресу **`http://localhost:8000/docs`** (корень `/` автоматически перенаправляет на `/docs`). Проверка эндпоинтов:
```bash
# Проверка готовности
curl http://localhost:8000/health

# Локальная классификация стиля (работает без LLM_API_KEY)
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"question": "В саду за старой усадьбой тихо осыпались липы; доктор пил остывший чай на террасе.", "k": 3}'

# Классификация + стилизованная генерация
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Какая прекрасная ночь над Невой...", "k": 3, "temperature": 0.7}'

# Многошаговый ИИ-агент (поддерживаются approach: "native", "json", "framework")
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "Узнай F1-score Чехова А и Достоевского Ф и вычисли разницу", "approach": "native"}'
```

### 3. Демонстрация MLOps и реестра моделей (MLflow)
Запустите серию экспериментов и откройте веб-интерфейс MLflow:
```bash
# Быстрая сетка из 4 запусков (или без --quick для полной сетки из 9 запусков; --synthetic для автономного режима)
python -m src.train_mlflow --quick
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Веб-интерфейс будет доступен по адресу **`http://127.0.0.1:5000`** (эксперимент **`doc-classifier`**, вкладка *Artifacts* с графиками `accuracy_chart.png` и `run_config.json`, а также *Models* → **`literary-style-classifier`** со статусом **`Staging`** и алиасом `@staging`).

### 4. Демонстрация 3 подходов ИИ-агентов
```bash
# Автономный демо-прогон с замоканной LLM и реальным классификатором + запись MLflow Traces:
python agents_comparison.py --mock --trace

# Живой прогон с реальным LLM API из .env + запись MLflow Traces:
python agents_comparison.py --trace
```
В консоли отобразится пошаговая трассировка вызовов инструментов (`classify_style` -> `author_metrics` -> `author_metrics` -> `calc`) и сводная таблица сравнения для **Native Tool Calling**, **ручного JSON-агента** и **LangGraph**, а при `--trace` полные спаны будут доступны в MLflow UI (`doc-classifier-agents`).

### 5. Проверка качества (линтер, автотесты, ноутбуки и Docker)
```bash
# Линтер Ruff и автономные тесты (соответствует GitHub Actions CI)
ruff check src tests agents_comparison.py --line-length 105
CLASSIFIER_BACKEND=baseline pytest tests -v --cov=src --cov-report=term-missing -k "not test_live_e2e and not test_baseline_and_lora"

# Полный прогон в Docker
docker build -t literary-style-assistant .
docker run --rm -p 8000:8000 -e LLM_API_KEY="Ваш_API_Ключ_LLM" literary-style-assistant
```
Воспроизведение полного исследовательского цикла в Jupyter-ноутбуках (`notebooks/`):
`01_eda_and_cleaning.ipynb` → `02_tfidf_experiments.ipynb` → `03_pytorch_baseline.ipynb` → `04_lora_finetuning.ipynb`.


Дополнительные материалы в репозитории:
* [REPORT.md](REPORT.md) — аналитический отчёт по экспериментам, метрикам, латентности и покрытию тестами.
* [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — простое объяснение проекта «на пальцах» и подробное руководство пользователя.
* [docs/CODEBASE_ARCHITECTURE.md](docs/CODEBASE_ARCHITECTURE.md) — карта кода с разбором каждого модуля и графами зависимостей.
* [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) — полная инженерная спецификация API, MLOps, ИИ-агентов и эксплуатации.
