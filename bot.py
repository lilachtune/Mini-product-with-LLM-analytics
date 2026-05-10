import os
import io
import logging
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from telegram import Update, InputFile
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes,
)

from agent import run_analysis_agent
from security import check_prompt_injection, sanitize_user_context

load_dotenv()

logging.basicConfig(
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

groq_client = Groq(api_key=GROQ_API_KEY)

user_contexts: dict[int, str] = {}

MAX_FILE_SIZE_MB = 10
SUPPORTED_EXTENSIONS = ('.csv', '.xlsx', '.xls')


def _load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    file_io = io.BytesIO(file_bytes)

    if filename.endswith('.csv'):
        for enc in ('utf-8', 'utf-8-sig', 'cp1251', 'latin-1'):
            try:
                file_io.seek(0)
                df = pd.read_csv(file_io, encoding=enc)
                for col in df.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except Exception:
                            pass
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError("Не удалось определить кодировку CSV-файла.")
    else:
        df = pd.read_excel(file_io)
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass
        return df


def _split_message(text: str, max_len: int = 4096) -> list[str]:
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        parts.append(text[:max_len])
        text = text[max_len:]
    return parts


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Команды:\n"
        "/context — задать контекст анализа\n"
        "/mycontext — посмотреть текущий контекст\n"
        "/clear — очистить контекст\n"
        "/help — справка"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Справка по боту\n\n"
        "Поддерживаемые форматы: CSV, XLSX, XLS (до 10 МБ)\n\n"
        "Как работает анализ:\n"
        "LLM-агент получает датасет и самостоятельно пишет Python-код для его исследования. "
        "Код реально выполняется — агент видит результаты и продолжает анализ. "
        "Это не перефразирование, а настоящий агентный подход.\n\n"
        "Пример контекста:\n"
        "/context Это данные интернет-магазина за 2023 год. Меня интересует сезонность спроса, "
        "топ-продукты и сравнение каналов продаж. Обрати внимание на аномальные недели."
    )


async def cmd_set_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id

    if not context.args:
        await update.message.reply_text(
            "Использование: /context <инструкции для анализа>\n\n"
            "Пример: /context Это продажи магазина. Найди сезонность и топ-категории."
        )
        return

    raw_context = " ".join(context.args)

    is_safe, reason = check_prompt_injection(raw_context)
    if not is_safe:
        logger.warning(f"Попытка инъекции в /context от пользователя {user_id}: {reason}")
        await update.message.reply_text(
            "Контекст содержит недопустимые инструкции и был отклонён.\n"
            "Опиши задачу своими словами: что это за данные и на что обратить внимание."
        )
        return

    clean_context = sanitize_user_context(raw_context)
    user_contexts[user_id] = clean_context
    await update.message.reply_text(f"Контекст сохранён:\n{clean_context}")


async def cmd_my_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    ctx = user_contexts.get(user_id)
    if ctx:
        await update.message.reply_text(f"Текущий контекст:\n{ctx}")
    else:
        await update.message.reply_text("Контекст не задан. Используй /context <текст>.")


async def cmd_clear_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    user_contexts.pop(user_id, None)
    await update.message.reply_text("Контекст очищен.")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    document = update.message.document
    caption: str = update.message.caption or ""

    filename = document.file_name or ""
    if not any(filename.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        await update.message.reply_text(
            "Неподдерживаемый формат. Отправь файл CSV, XLSX или XLS."
        )
        return

    if document.file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        await update.message.reply_text(f"Файл слишком большой. Максимум: {MAX_FILE_SIZE_MB} МБ.")
        return

    if caption:
        is_safe, _ = check_prompt_injection(caption)
        if not is_safe:
            logger.warning(f"Попытка инъекции в подписи от пользователя {user_id}")
            await update.message.reply_text(
                "Подпись к файлу содержит недопустимые инструкции и была проигнорирована."
            )
            caption = ""

    saved_context = user_contexts.get(user_id, "")
    parts = [p for p in [sanitize_user_context(caption), saved_context] if p]
    final_context = "\n".join(parts)

    await update.message.reply_chat_action(ChatAction.TYPING)
    status_msg = await update.message.reply_text(
        "Файл получен. Запускаю AI-агент...\n"
        "Анализ займёт около 1-2 минут.\n\n"
        "Агент пишет и выполняет код итеративно."
    )

    try:
        tg_file = await document.get_file()
        file_bytes = bytes(await tg_file.download_as_bytearray())

        df = _load_dataframe(file_bytes, filename.lower())

        if df.empty:
            await status_msg.edit_text("Датасет пустой — нечего анализировать.")
            return

        await status_msg.edit_text(
            f"Датасет загружен: {df.shape[0]:,} строк x {df.shape[1]} столбцов\n"
            f"AI-агент начал анализ... (модель: {GROQ_MODEL})"
        )

        report, figures = run_analysis_agent(df, final_context, groq_client, model=GROQ_MODEL)

        if figures:
            await update.message.reply_text(f"Агент создал {len(figures)} визуализаций:")
            for i, fig_buf in enumerate(figures, start=1):
                fig_buf.seek(0)
                await update.message.reply_photo(
                    photo=InputFile(fig_buf, filename=f"chart_{i}.png"),
                    caption=f"График {i}/{len(figures)}",
                )

        await update.message.reply_text("Аналитический отчёт:")
        for part in _split_message(report):
            await update.message.reply_text(part)

        await status_msg.edit_text("Анализ завершён.")

    except ValueError as e:
        logger.warning(f"Ошибка загрузки файла для пользователя {user_id}: {e}")
        await status_msg.edit_text(f"Ошибка загрузки файла: {e}")
    except Exception as e:
        logger.error(f"Непредвиденная ошибка для пользователя {user_id}: {e}", exc_info=True)
        await status_msg.edit_text(
            "Произошла ошибка при анализе. Попробуй снова или проверь формат данных."
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Отправь CSV или Excel-файл для анализа.\n"
        "Чтобы задать контекст — используй /context <инструкции>."
    )


def main() -> None:
    logger.info("Запуск Analytics Bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("context", cmd_set_context))
    app.add_handler(CommandHandler("mycontext", cmd_my_context))
    app.add_handler(CommandHandler("clear", cmd_clear_context))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Бот запущен. Для остановки нажми Ctrl+C.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()