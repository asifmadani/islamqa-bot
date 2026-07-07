#!/usr/bin/env python3
"""
Darul Ilm Q&A Bot
Flow:
  1. Website → Telegram group mein question aata hai (existing JS)
  2. Sheikh audio reply kare us question par
  3. Bot Groq Whisper se Urdu transcript banata hai
  4. Sheikh Telegram mein review/edit/approve kare
  5. Approve → GitHub API se qa.html update hoti hai → site deploy
"""

import os, re, base64, logging, asyncio, tempfile
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes,
)
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
TG_TOKEN    = os.environ["TG_TOKEN"]
TG_CHAT_ID  = int(os.environ["TG_CHAT_ID"])
GROQ_KEY    = os.environ["GROQ_KEY"]
GH_TOKEN    = os.environ["GH_TOKEN"]
GH_REPO     = "asifmadani/asifmadani.github.io"
WEBHOOK_URL = os.environ["WEBHOOK_URL"]   # https://your-app.onrender.com (no trailing slash)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

groq_client = Groq(api_key=GROQ_KEY)

# ── In-memory state ───────────────────────────────────────────────────────────
# pending_review[key] = {name, email, question, answer_draft}
pending_review: dict[str, dict] = {}
# editing_state[telegram_user_id] = key
editing_state: dict[int, str] = {}

# ── Telegram app ──────────────────────────────────────────────────────────────
ptb_app = Application.builder().token(TG_TOKEN).build()

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_question_msg(text: str) -> dict:
    """Parse the structured question text posted by the website JS."""
    result = {}
    m = re.search(r"👤 Name: (.+)", text)
    if m: result["name"] = m.group(1).strip()
    m = re.search(r"📂 Category: (.+)", text)
    if m: result["category"] = m.group(1).strip()
    m = re.search(r"📧 Email: (.+)", text)
    if m: result["email"] = m.group(1).strip()
    # Question is everything after the last separator line
    m = re.search(r"❓ Question:\n(.+)", text, re.DOTALL)
    if m: result["question"] = m.group(1).strip()
    return result


def transcribe_sync(audio_bytes: bytes, suffix: str) -> str:
    """Synchronous Groq Whisper call (runs in thread executor)."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        with open(path, "rb") as f:
            resp = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(os.path.basename(path), f),
                language="ur",
                response_format="text",
            )
        return resp if isinstance(resp, str) else resp.text
    finally:
        os.unlink(path)


async def publish_to_github(name: str, question: str, answer: str) -> bool:
    """Insert new Q&A item into qa.html via GitHub API and trigger deploy."""
    headers = {
        "Authorization": f"token {GH_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"https://api.github.com/repos/{GH_REPO}/contents/qa.html"

    # New accordion block to insert
    block = (
        '\n        <div class="qa-item">\n'
        '          <div class="qa-question">\n'
        f'            <h3>{question}</h3>\n'
        '            <span class="qa-toggle">+</span>\n'
        '          </div>\n'
        '          <div class="qa-answer">\n'
        f'            <p>{answer}</p>\n'
        f'            <p style="font-size:0.82rem;color:var(--text-light);margin-top:0.5rem;">'
        f'— {name} ka sawal</p>\n'
        '          </div>\n'
        '        </div>'
    )

    marker = "<h2>Published Answers</h2>"

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=headers)
        if not r.is_success:
            log.error("GitHub GET %s: %s", r.status_code, r.text[:200])
            return False

        data    = r.json()
        sha     = data["sha"]
        content = base64.b64decode(data["content"]).decode("utf-8")

        if marker not in content:
            log.error("Marker '%s' not found in qa.html", marker)
            return False

        updated = content.replace(marker, marker + block, 1)
        encoded = base64.b64encode(updated.encode("utf-8")).decode("utf-8")

        r2 = await client.put(url, headers=headers, json={
            "message": f"Q&A: {question[:60]}",
            "content": encoded,
            "sha": sha,
            "committer": {
                "name": "Darul Ilm Bot",
                "email": "bot@darulilm.com",
            },
        })
        if not r2.is_success:
            log.error("GitHub PUT %s: %s", r2.status_code, r2.text[:200])
            return False

        log.info("Published Q&A to GitHub ✓")
        return True


def review_keyboard(key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve & Publish", callback_data=f"approve|{key}"),
            InlineKeyboardButton("✏️ Edit",              callback_data=f"edit|{key}"),
        ],
        [InlineKeyboardButton("❌ Discard", callback_data=f"discard|{key}")],
    ])


def review_text(q: dict) -> str:
    return (
        "📝 *Transcription — Review karo:*\n\n"
        f"❓ *Sawal:*\n{q['question'][:300]}\n\n"
        f"💬 *Jawab (Draft):*\n{q['answer_draft']}\n\n"
        "_Approve karein, Edit karein, ya Discard karein._"
    )


# ── Telegram Handlers ─────────────────────────────────────────────────────────

async def on_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Detect audio reply to a question message → transcribe → show review."""
    msg = update.message
    if not msg or msg.chat_id != TG_CHAT_ID:
        return
    if not msg.reply_to_message:
        return   # not a reply — ignore

    orig = msg.reply_to_message.text or ""
    if "❓ Question:" not in orig:
        return   # not a question message — ignore

    fields = parse_question_msg(orig)
    if not fields.get("question"):
        return

    proc = await ctx.bot.send_message(
        chat_id=TG_CHAT_ID,
        text="⏳ Audio sun raha hoon, Urdu transcript ban rahi hai...",
        reply_to_message_id=msg.message_id,
    )

    try:
        voice_obj = msg.voice or msg.audio
        tg_file   = await ctx.bot.get_file(voice_obj.file_id)
        audio_bytes = bytes(await tg_file.download_as_bytearray())
        suffix    = ".ogg" if msg.voice else ".mp3"

        loop   = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None, transcribe_sync, audio_bytes, suffix
        )
    except Exception as exc:
        log.exception("Transcription error")
        await ctx.bot.edit_message_text(
            chat_id=TG_CHAT_ID,
            message_id=proc.message_id,
            text=f"❌ Transcription fail hui:\n{exc}",
        )
        return

    key = str(proc.message_id)
    pending_review[key] = {
        "name":         fields.get("name", "Anonymous"),
        "email":        fields.get("email", ""),
        "question":     fields["question"],
        "answer_draft": answer,
    }

    await ctx.bot.edit_message_text(
        chat_id=TG_CHAT_ID,
        message_id=proc.message_id,
        text=review_text(pending_review[key]),
        parse_mode="Markdown",
        reply_markup=review_keyboard(key),
    )


async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle Approve / Edit / Discard button presses."""
    query = update.callback_query
    await query.answer()

    action, key = query.data.split("|", 1)
    q = pending_review.get(key)
    if not q:
        await query.edit_message_text("⚠️ Session expire ho gaya. Dobara try karo.")
        return

    if action == "approve":
        await query.edit_message_text("⏳ Website par publish ho raha hai...")
        ok = await publish_to_github(q["name"], q["question"], q["answer_draft"])
        if ok:
            pending_review.pop(key, None)
            await query.edit_message_text(
                "✅ *Website par publish ho gaya!*\n\n"
                f"❓ {q['question'][:200]}\n\n"
                f"💬 {q['answer_draft'][:200]}\n\n"
                "🔗 https://asifmadani.github.io/qa.html",
                parse_mode="Markdown",
            )
        else:
            await query.edit_message_text(
                "❌ GitHub update fail hua. Retry karo.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔄 Retry",   callback_data=f"approve|{key}"),
                    InlineKeyboardButton("❌ Discard", callback_data=f"discard|{key}"),
                ]]),
            )

    elif action == "edit":
        editing_state[query.from_user.id] = key
        await query.edit_message_text(
            "✏️ *Edit mode:*\n\n"
            f"Purana jawab:\n_{q['answer_draft']}_\n\n"
            "Ab sahi jawab group mein type karke bhejo:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_edit|{key}")
            ]]),
        )

    elif action == "discard":
        pending_review.pop(key, None)
        await query.edit_message_text("🗑️ Jawab discard kar diya gaya.")

    elif action == "cancel_edit":
        editing_state.pop(query.from_user.id, None)
        await query.edit_message_text(
            review_text(q),
            parse_mode="Markdown",
            reply_markup=review_keyboard(key),
        )


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Capture edited answer text from Sheikh."""
    msg = update.message
    if not msg or msg.chat_id != TG_CHAT_ID:
        return
    user_id = msg.from_user.id
    if user_id not in editing_state:
        return

    key = editing_state.pop(user_id)
    if key not in pending_review:
        return

    pending_review[key]["answer_draft"] = msg.text

    await msg.reply_text(
        review_text(pending_review[key]),
        parse_mode="Markdown",
        reply_markup=review_keyboard(key),
    )


# ── FastAPI ───────────────────────────────────────────────────────────────────

async def keep_alive():
    """Ping self every 14 minutes so Render free tier doesn't sleep."""
    while True:
        await asyncio.sleep(14 * 60)
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                await c.get(WEBHOOK_URL + "/")
            log.info("Keep-alive ping sent")
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    ptb_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    ptb_app.add_handler(CallbackQueryHandler(on_callback))
    ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    await ptb_app.initialize()
    webhook_path = f"{WEBHOOK_URL}/webhook/{TG_TOKEN}"
    await ptb_app.bot.set_webhook(webhook_path)
    log.info("Webhook set → %s", webhook_path)
    await ptb_app.start()

    asyncio.create_task(keep_alive())

    yield
    await ptb_app.stop()
    await ptb_app.shutdown()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def health():
    return {"status": "ok", "bot": "Darul Ilm Q&A Bot ✓"}


@app.post(f"/webhook/{TG_TOKEN}")
async def tg_webhook(request: Request):
    body   = await request.json()
    update = Update.de_json(body, ptb_app.bot)
    await ptb_app.process_update(update)
    return {"ok": True}
