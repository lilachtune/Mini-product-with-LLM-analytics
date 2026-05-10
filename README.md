# Telegram-бот для агентного анализа данных с помощью LLM 

Продукт, который решает аналитическую задачу с помощью ИИ-агента. Реализовывал через Groq API, модель llama-3.3-70b-versatile.

## Архитектура

```
Пользователь (Telegram)
       │
       ▼
  bot.py  ←── Валидация + защита от prompt injection (security.py)
       │
       ▼
  agent.py ──► Groq API (llama-3.3-70b-versatile)
       │              │
       │    ┌─────────▼──────────┐
       │    │   Tool: execute_python  │  ← LLM сама пишет код
       │    └─────────┬──────────┘
       │              │
       │    ┌─────────▼──────────┐
       │    │  Реальное выполнение│  ← Sandbox exec() + matplotlib
       │    │  Python кода       │
       │    └─────────┬──────────┘
       │              │  результат
       │    ◄─────────┘  (агент итерирует до финального отчёта)
       │
       ▼
  Отчёт + Графики → Telegram
```



## Требования

- Python 3.10+
- Telegram Bot Token ([@BotFather](https://t.me/BotFather))
- Groq API Key ([console.groq.com](https://console.groq.com))

## Установка

```bash
# 1. Клонируй репозиторий

# 2. Создай виртуальное окружение
python -m venv venv
venv\Scripts\activate.bat     

# 3. Установи зависимости
pip install -r requirements.txt

# 4. Создай .env файл со след содержанием и заполни ключи
TELEGRAM_BOT_TOKEN=xxxxxxxxx

GROQ_API_KEY=xxxxxxxxxxxx

GROQ_MODEL=llama-3.3-70b-versatile

# 5. Запусти бота
python bot.py
```

### Тестовый датасет
---

В папке лежит `sales_2023_2024.csv` — данные продаж 
интернет-магазина электроники за 2 года (1800 заказов, 12 колонок).


## Защита от Prompt Injection

Реализована многоуровневая защита:

| Уровень | Где | Что проверяем |
|---------|-----|---------------|
| **L1** | `security.py` | Regex-паттерны в пользовательском тексте (15+ паттернов) |
| **L2** | `security.py` | Длина контекста (макс 2000 символов) |
| **L3** | `agent.py` | Проверка кода LLM перед exec() (запрет os, sys, socket...) |
| **L4** | `agent.py` | Restricted `__builtins__` в exec-окружении |
| **L5** | System prompt | Явное указание на изоляцию роли + маркировка user_context тегами |

**Контекст пользователя изолирован в промпте тегами `<user_context>...</user_context>`** — 
LLM воспринимает контекст как описание данных, а не системные команды.

---

## Структура проекта

```
analytics_bot/
├── bot.py              # Telegram-бот, обработчики
├── agent.py            # LLM-агент с tool calling, sandbox exec
├── security.py         # Защита от prompt injection
├── requirements.txt    # Необходимые для работы зависимости
├── .env                # Файл с ключами
├── sales_2023_2024.csv  # Тестовый датасет 
└──demonstration_of_working.mp4 # Демонстрация работы проекта

```

## Колонки тестового датасета

| Колонка | Тип | Описание |
|---------|-----|----------|
| `order_date` | date | Дата заказа |
| `product` | str | Название товара |
| `category` | str | Категория (5 штук) |
| `region` | str | Регион (6 городов) |
| `channel` | str | Канал продаж (4 штуки) |
| `customer_segment` | str | Сегмент клиента |
| `quantity` | int | Количество единиц |
| `unit_price` | int | Цена за единицу (руб) |
| `revenue` | int | Выручка (руб) |
| `profit` | int | Прибыль (руб) |
| `rating` | float | Оценка (1-5, ~3% пропусков) |
| `is_returned` | int | Возврат (0/1) |

## Что агент сделает с этим датасетом

1. `execute_python` → исследует структуру, типы, пропуски
2. `execute_python` → дескриптивная статистика числовых колонок
3. `execute_python` → временной ряд выручки по месяцам (график)
4. `execute_python` → анализ по категориям (bar chart)
5. `execute_python` → корреляционная матрица (heatmap)
6. `execute_python` → анализ каналов и регионов
7. `execute_python` → анализ возвратов и рейтингов
8. Финальный отчёт на русском с инсайтами и рекомендациями

---