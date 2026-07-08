#!/usr/bin/env python3
"""
Darul Ilm Content Management Bot
Handles:
  - Q&A: audio reply → Urdu transcript → approve → publish to qa.html
  - #video: YouTube link → approve → publish to videos.html
  - #maqalah: title + text → approve → publish to maqalah.html
  - #research: title + description [+ PDF] → approve → publish to research.html
  - #book: title + language + PDF → approve → upload PDF + publish to books.html
"""

import os, re, base64, logging, asyncio, tempfile
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, MessageHandler, CallbackQueryHandler, CommandHandler,
    filters, ContextTypes,
)
from groq import Groq

# ── Config ─────────────────────────────────────────────────────────────────────
TG_TOKEN    = os.environ["TG_TOKEN"]
TG_CHAT_ID  = int(os.environ["TG_CHAT_ID"])
GROQ_KEY    = os.environ["GROQ_KEY"]
GH_TOKEN    = os.environ["GH_TOKEN"]
GH_REPO     = "asifmadani/asifmadani.github.io"
WEBHOOK_URL = os.environ["WEBHOOK_URL"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

groq_client = Groq(api_key=GROQ_KEY)

# ── In-memory state ────────────────────────────────────────────────────────────
pending_review: dict[str, dict] = {}
editing_state: dict[int, str] = {}
manage_cache: dict[int, dict] = {}   # user_id → {op, page, blocks}
edit_draft:   dict[int, dict] = {}   # user_id → {page, idx, old_block}

# ── Telegram app ───────────────────────────────────────────────────────────────
ptb_app = Application.builder().token(TG_TOKEN).build()


# ══════════════════════════════════════════════════════════════════════════════
# PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_question_msg(text: str) -> dict:
    result = {}
    m = re.search(r"👤 Name: (.+)", text)
    if m: result["name"] = m.group(1).strip()
    m = re.search(r"📂 Category: (.+)", text)
    if m: result["category"] = m.group(1).strip()
    m = re.search(r"📧 Email: (.+)", text)
    if m: result["email"] = m.group(1).strip()
    m = re.search(r"❓ Question:\n(.+)", text, re.DOTALL)
    if m: result["question"] = m.group(1).strip()
    return result


def extract_youtube_id(text: str) -> str | None:
    m = re.search(
        r'(?:youtube\.com/watch\?[^"]*v=|youtu\.be/|youtube\.com/(?:shorts|live|embed)/)([A-Za-z0-9_-]{11})',
        text,
    )
    return m.group(1) if m else None


def parse_hashtag_msg(text: str) -> dict | None:
    tag_m = re.search(r'#(video|maqalah|tafseer|tashreeh|research|book)', text, re.IGNORECASE)
    if not tag_m:
        return None

    content_type = tag_m.group(1).lower()
    clean = re.sub(r'#\w+\s*', '', text).strip()

    title_m = re.search(r'(?:Title|عنوان):\s*(.+)', clean, re.IGNORECASE)
    desc_m  = re.search(r'(?:Description|تفصیل|Desc|Abstract):\s*(.+)', clean, re.IGNORECASE | re.DOTALL)
    lang_m  = re.search(r'(?:Language|زبان|Lang):\s*(\S+)', clean, re.IGNORECASE)

    title       = title_m.group(1).strip() if title_m else ""
    description = desc_m.group(1).strip()  if desc_m  else ""
    language    = lang_m.group(1).strip()  if lang_m  else "Urdu"

    if not title:
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        if lines:
            title = lines[0]
            if not description and len(lines) > 1:
                description = '\n'.join(lines[1:]).strip()

    result = {
        "type": content_type,
        "title": title,
        "description": description,
    }

    if content_type == "video":
        result["youtube_id"] = extract_youtube_id(text) or ""

    if content_type == "book":
        result["language"] = language

    return result


def transcribe_sync(audio_bytes: bytes, suffix: str) -> str:
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


# ══════════════════════════════════════════════════════════════════════════════
# GITHUB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_GH_HEADERS = lambda: {
    "Authorization": f"token {GH_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


async def gh_get_file(path: str) -> tuple[str, str] | tuple[None, None]:
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(url, headers=_GH_HEADERS())
        if not r.is_success:
            log.error("GH GET %s → %s", path, r.status_code)
            return None, None
        d = r.json()
        return d["sha"], base64.b64decode(d["content"]).decode("utf-8")


async def gh_put_file(path: str, sha: str, content: str, msg: str) -> bool:
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
    body = {
        "message": msg,
        "content": base64.b64encode(content.encode()).decode(),
        "sha": sha,
        "committer": {"name": "Darul Ilm Bot", "email": "bot@darulilm.com"},
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.put(url, headers=_GH_HEADERS(), json=body)
        if not r.is_success:
            log.error("GH PUT %s → %s %s", path, r.status_code, r.text[:200])
            return False
        return True


async def gh_upload_binary(path: str, file_bytes: bytes, msg: str) -> bool:
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path}"
    body: dict = {
        "message": msg,
        "content": base64.b64encode(file_bytes).decode(),
        "committer": {"name": "Darul Ilm Bot", "email": "bot@darulilm.com"},
    }
    async with httpx.AsyncClient(timeout=120) as c:
        r_get = await c.get(url, headers=_GH_HEADERS())
        if r_get.is_success:
            body["sha"] = r_get.json()["sha"]
        r = await c.put(url, headers=_GH_HEADERS(), json=body)
        if not r.is_success:
            log.error("GH PUT binary %s → %s %s", path, r.status_code, r.text[:200])
            return False
        return True


# ══════════════════════════════════════════════════════════════════════════════
# PUBLISH FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

async def publish_qa(name: str, question: str, answer: str) -> bool:
    sha, html = await gh_get_file("qa.html")
    if not sha:
        return False
    marker = "<h2>Published Answers</h2>"
    if marker not in html:
        log.error("BOT:qa marker missing")
        return False
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
    return await gh_put_file("qa.html", sha, html.replace(marker, marker + block, 1), f"Q&A: {question[:60]}")


async def publish_video(title: str, description: str, youtube_id: str) -> bool:
    sha, html = await gh_get_file("videos.html")
    if not sha:
        return False
    marker = "<!-- BOT:video -->"
    if marker not in html:
        log.error("BOT:video marker missing")
        return False
    block = (
        f'\n      <div class="video-card">\n'
        f'        <div class="video-thumb">\n'
        f'          <iframe src="https://www.youtube.com/embed/{youtube_id}" '
        f'allowfullscreen title="{title}"></iframe>\n'
        f'        </div>\n'
        f'        <div class="video-info">\n'
        f'          <h3>{title}</h3>\n'
        f'          <p>{description}</p>\n'
        f'        </div>\n'
        f'      </div>\n'
    )
    return await gh_put_file("videos.html", sha, html.replace(marker, marker + block, 1), f"Video: {title[:60]}")


async def publish_maqalah(title: str, description: str) -> bool:
    sha, html = await gh_get_file("maqalah.html")
    if not sha:
        return False
    marker = "<!-- BOT:maqalah -->"
    if marker not in html:
        log.error("BOT:maqalah marker missing")
        return False
    block = (
        f'\n      <div class="topic-card">\n'
        f'        <h3>{title}</h3>\n'
        f'        <p>{description}</p>\n'
        f'      </div>\n'
    )
    return await gh_put_file("maqalah.html", sha, html.replace(marker, marker + block, 1), f"Maqalah: {title[:60]}")


async def publish_tafseer(title: str, description: str) -> bool:
    sha, html = await gh_get_file("tafseer.html")
    if not sha:
        return False
    marker = "<!-- BOT:tafseer -->"
    if marker not in html:
        log.error("BOT:tafseer marker missing")
        return False
    block = (
        f'\n      <div class="topic-card">\n'
        f'        <h3>{title}</h3>\n'
        f'        <p>{description}</p>\n'
        f'      </div>\n'
    )
    return await gh_put_file("tafseer.html", sha, html.replace(marker, block + marker, 1), f"Tafseer: {title[:60]}")


async def publish_tashreeh(title: str, description: str) -> bool:
    sha, html = await gh_get_file("tashreeh.html")
    if not sha:
        return False
    marker = "<!-- BOT:tashreeh -->"
    if marker not in html:
        log.error("BOT:tashreeh marker missing")
        return False
    block = (
        f'\n      <div class="topic-card">\n'
        f'        <h3>{title}</h3>\n'
        f'        <p>{description}</p>\n'
        f'      </div>\n'
    )
    return await gh_put_file("tashreeh.html", sha, html.replace(marker, block + marker, 1), f"Tashreeh: {title[:60]}")


async def publish_research(title: str, description: str, pdf_filename: str = "") -> bool:
    sha, html = await gh_get_file("research.html")
    if not sha:
        return False
    marker = "<!-- BOT:research -->"
    if marker not in html:
        log.error("BOT:research marker missing")
        return False
    year = datetime.now().strftime("%Y")
    dl = (f'\n          <a href="files/{pdf_filename}" download class="btn-download" '
          f'target="_blank">⬇ Download PDF</a>') if pdf_filename else ""
    block = (
        f'\n      <div class="pub-card">\n'
        f'        <div class="pub-icon">📄</div>\n'
        f'        <div class="pub-info">\n'
        f'          <h3>{title}</h3>\n'
        f'          <p>{description}</p>\n'
        f'          <span style="color:var(--text-light);font-size:0.82rem;">'
        f'{year} · Asif Jamiee Madani Hafizahullah</span>{dl}\n'
        f'        </div>\n'
        f'      </div>\n'
    )
    return await gh_put_file("research.html", sha, html.replace(marker, marker + block, 1), f"Research: {title[:60]}")


async def publish_book(title: str, description: str, language: str, pdf_filename: str) -> bool:
    sha, html = await gh_get_file("books.html")
    if not sha:
        return False
    marker = "<!-- BOT:books -->"
    if marker not in html:
        log.error("BOT:books marker missing")
        return False
    icon = "📗" if language.lower() in ("english", "en") else "📘"
    block = (
        f'\n      <div class="pub-card">\n'
        f'        <div class="pub-icon">{icon}</div>\n'
        f'        <div class="pub-info">\n'
        f'          <h3>{title}</h3>\n'
        f'          <p>{description}</p>\n'
        f'          <span style="color:var(--text-light);font-size:0.82rem;">'
        f'Authored by Asif Jamiee Madani Hafizahullah</span><br/><br/>\n'
        f'          <a href="files/{pdf_filename}" download class="btn-download" '
        f'target="_blank">⬇ Download PDF</a>\n'
        f'        </div>\n'
        f'      </div>\n'
    )
    return await gh_put_file("books.html", sha, html.replace(marker, marker + block, 1), f"Book: {title[:60]}")


# ══════════════════════════════════════════════════════════════════════════════
# REVIEW UI
# ══════════════════════════════════════════════════════════════════════════════

def review_keyboard(key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Approve & Publish", callback_data=f"approve|{key}"),
            InlineKeyboardButton("✏️ Edit",              callback_data=f"edit|{key}"),
        ],
        [InlineKeyboardButton("❌ Discard", callback_data=f"discard|{key}")],
    ])


def review_text(q: dict) -> str:
    t = q.get("type", "qa")
    if t == "qa":
        return (
            "📝 *Transcription — Review karo:*\n\n"
            f"❓ *Sawal:* {q['question'][:300]}\n\n"
            f"💬 *Jawab (Draft):*\n{q['answer_draft']}\n\n"
            "_Approve, Edit, ya Discard karein._"
        )
    elif t == "video":
        yt = q.get("youtube_id", "")
        return (
            "🎬 *Video — Review karo:*\n\n"
            f"📌 *Title:* {q.get('title','')}\n"
            f"📝 *Description:* {q.get('description','')}\n"
            f"🔗 *YouTube ID:* `{yt}`\n"
            f"🖼 Preview: youtube.com/watch?v={yt}\n\n"
            "_Approve karo to Videos page par add ho jaega._"
        )
    elif t == "maqalah":
        return (
            "📚 *Maqalah — Review karo:*\n\n"
            f"📌 *Title:* {q.get('title','')}\n"
            f"📝 *Content:* {q.get('description','')[:400]}\n\n"
            "_Approve karo to Maqalah page par add ho jaega._"
        )
    elif t == "tafseer":
        return (
            "📜 *Tafseer — Review karo:*\n\n"
            f"📌 *Title:* {q.get('title','')}\n"
            f"📝 *Content:* {q.get('description','')[:400]}\n\n"
            "_Approve karo to Tafseer page par add ho jaega._"
        )
    elif t == "tashreeh":
        return (
            "📋 *Tashreeh — Review karo:*\n\n"
            f"📌 *Title:* {q.get('title','')}\n"
            f"📝 *Content:* {q.get('description','')[:400]}\n\n"
            "_Approve karo to Tashreeh page par add ho jaega._"
        )
    elif t == "research":
        pdf = f"📎 *PDF:* `{q.get('pdf_filename','')}`\n" if q.get("pdf_filename") else "📎 *PDF:* nahi\n"
        return (
            "🔬 *Research — Review karo:*\n\n"
            f"📌 *Title:* {q.get('title','')}\n"
            f"📝 *Description:* {q.get('description','')[:300]}\n"
            f"{pdf}\n"
            "_Approve karo to Research page par add ho jaega._"
        )
    elif t == "book":
        return (
            "📖 *Book — Review karo:*\n\n"
            f"📌 *Title:* {q.get('title','')}\n"
            f"🌐 *Language:* {q.get('language','')}\n"
            f"📝 *Description:* {q.get('description','')[:300]}\n"
            f"📎 *PDF:* `{q.get('pdf_filename','')}`\n\n"
            "_Approve karo to Books page par add ho jaega._"
        )
    return "Unknown content type."


PAGE_URLS = {
    "qa":       "https://asifmadani.github.io/qa.html",
    "video":    "https://asifmadani.github.io/videos.html",
    "maqalah":  "https://asifmadani.github.io/maqalah.html",
    "tafseer":  "https://asifmadani.github.io/tafseer.html",
    "tashreeh": "https://asifmadani.github.io/tashreeh.html",
    "research": "https://asifmadani.github.io/research.html",
    "book":     "https://asifmadani.github.io/books.html",
}


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

async def on_voice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Sheikh audio reply to a question → transcribe → review."""
    msg = update.message
    if not msg or msg.chat_id != TG_CHAT_ID:
        return
    if not msg.reply_to_message:
        return

    orig = msg.reply_to_message.text or ""
    if "❓ Question:" not in orig:
        return

    fields = parse_question_msg(orig)
    if not fields.get("question"):
        return

    proc = await ctx.bot.send_message(
        chat_id=TG_CHAT_ID,
        text="⏳ Audio sun raha hoon, Urdu transcript ban rahi hai...",
        reply_to_message_id=msg.message_id,
    )
    try:
        voice_obj   = msg.voice or msg.audio
        tg_file     = await ctx.bot.get_file(voice_obj.file_id)
        audio_bytes = bytes(await tg_file.download_as_bytearray())
        suffix      = ".ogg" if msg.voice else ".mp3"
        loop        = asyncio.get_event_loop()
        answer      = await loop.run_in_executor(None, transcribe_sync, audio_bytes, suffix)
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
        "type":         "qa",
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


async def on_hashtag_msg(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle #video / #maqalah / #research / #book messages."""
    try:
        msg = update.message
        if not msg or msg.chat_id != TG_CHAT_ID:
            return

        # Don't intercept if user is in edit mode
        if msg.from_user and msg.from_user.id in editing_state:
            return

        text = (msg.text or "") + " " + (msg.caption or "")
        if not re.search(r'#(video|maqalah|tafseer|tashreeh|research|book)', text, re.IGNORECASE):
            return

        log.info("Hashtag detected in: %s", text[:100])
        parsed = parse_hashtag_msg(text)
        if not parsed:
            log.error("parse_hashtag_msg returned None for: %s", text[:100])
            return
        log.info("Parsed: type=%s title=%s", parsed.get('type'), parsed.get('title'))
    except Exception as exc:
        log.exception("on_hashtag_msg early error")
        await ctx.bot.send_message(chat_id=TG_CHAT_ID, text=f"❌ Bot error (early): {exc}")
        return

    content_type = parsed["type"]

    # Download PDF for book/research
    if content_type in ("book", "research") and msg.document:
        doc = msg.document
        is_pdf = (doc.mime_type == "application/pdf") or (doc.file_name or "").endswith(".pdf")
        if is_pdf:
            proc = await ctx.bot.send_message(
                chat_id=TG_CHAT_ID,
                text="⏳ PDF download ho raha hai...",
                reply_to_message_id=msg.message_id,
            )
            try:
                tg_file   = await ctx.bot.get_file(doc.file_id)
                pdf_bytes = bytes(await tg_file.download_as_bytearray())
                safe      = re.sub(r'[^a-zA-Z0-9_-]', '-', parsed["title"] or "file")[:40]
                pdf_fn    = f"{safe}-{msg.message_id}.pdf"
                parsed["pdf_bytes"]    = pdf_bytes
                parsed["pdf_filename"] = pdf_fn
                await ctx.bot.delete_message(chat_id=TG_CHAT_ID, message_id=proc.message_id)
            except Exception as exc:
                log.exception("PDF download error")
                await ctx.bot.edit_message_text(
                    chat_id=TG_CHAT_ID, message_id=proc.message_id,
                    text=f"❌ PDF download fail hua:\n{exc}",
                )
                return

    # Validation
    if content_type == "book" and not parsed.get("pdf_filename"):
        await ctx.bot.send_message(
            chat_id=TG_CHAT_ID,
            text=(
                "❌ *Book upload ke liye PDF file zaroori hai!*\n\n"
                "PDF attach karke dobara bhejein:\n"
                "`#book`\n`Title: Book ka naam`\n`Language: Urdu`\n`[PDF file attach]`"
            ),
            parse_mode="Markdown",
            reply_to_message_id=msg.message_id,
        )
        return

    if content_type == "video" and not parsed.get("youtube_id"):
        await ctx.bot.send_message(
            chat_id=TG_CHAT_ID,
            text=(
                "❌ *YouTube link nahi mili!*\n\n"
                "YouTube URL ke saath dobara bhejein:\n"
                "`#video`\n`Title: Video ka naam`\n`https://youtube.com/watch?v=...`"
            ),
            parse_mode="Markdown",
            reply_to_message_id=msg.message_id,
        )
        return

    try:
        key = str(msg.message_id)
        pending_review[key] = parsed

        await ctx.bot.send_message(
            chat_id=TG_CHAT_ID,
            text=review_text(parsed),
            parse_mode="Markdown",
            reply_markup=review_keyboard(key),
            reply_to_message_id=msg.message_id,
        )
    except Exception as exc:
        log.exception("on_hashtag_msg send error")
        await ctx.bot.send_message(
            chat_id=TG_CHAT_ID,
            text=f"❌ Bot error (send): {exc}\n\nContent type: {parsed.get('type')}\nTitle: {parsed.get('title','')[:50]}",
            reply_to_message_id=msg.message_id,
        )


async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Approve / Edit / Discard button handler + delete/edit management."""
    query = update.callback_query
    await query.answer()

    # Route manage callbacks to separate handler
    if query.data.startswith("mgr|"):
        await handle_manage_callback(query, ctx)
        return

    action, key = query.data.split("|", 1)
    q = pending_review.get(key)
    if not q:
        await query.edit_message_text("⚠️ Session expire ho gaya. Dobara try karo.")
        return

    content_type = q.get("type", "qa")

    if action == "approve":
        await query.edit_message_text("⏳ Website par publish ho raha hai...")
        ok = False

        if content_type == "qa":
            ok = await publish_qa(q["name"], q["question"], q["answer_draft"])

        elif content_type == "video":
            ok = await publish_video(q["title"], q.get("description", ""), q["youtube_id"])

        elif content_type == "maqalah":
            ok = await publish_maqalah(q["title"], q.get("description", ""))

        elif content_type == "tafseer":
            ok = await publish_tafseer(q["title"], q.get("description", ""))

        elif content_type == "tashreeh":
            ok = await publish_tashreeh(q["title"], q.get("description", ""))

        elif content_type == "research":
            pdf_fn = q.get("pdf_filename", "")
            if q.get("pdf_bytes"):
                if not await gh_upload_binary(f"files/{pdf_fn}", q["pdf_bytes"], f"PDF: {q['title'][:50]}"):
                    await query.edit_message_text(
                        "❌ PDF upload fail hua. Retry karo.",
                        reply_markup=InlineKeyboardMarkup([[
                            InlineKeyboardButton("🔄 Retry",   callback_data=f"approve|{key}"),
                            InlineKeyboardButton("❌ Discard", callback_data=f"discard|{key}"),
                        ]]),
                    )
                    return
            ok = await publish_research(q["title"], q.get("description", ""), pdf_fn)

        elif content_type == "book":
            pdf_fn = q.get("pdf_filename", "")
            if not await gh_upload_binary(f"files/{pdf_fn}", q["pdf_bytes"], f"Book: {q['title'][:50]}"):
                await query.edit_message_text(
                    "❌ PDF upload fail hua. Retry karo.",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("🔄 Retry",   callback_data=f"approve|{key}"),
                        InlineKeyboardButton("❌ Discard", callback_data=f"discard|{key}"),
                    ]]),
                )
                return
            ok = await publish_book(q["title"], q.get("description", ""), q.get("language", "Urdu"), pdf_fn)

        if ok:
            pending_review.pop(key, None)
            title_preview = q.get("title") or q.get("question", "")
            await query.edit_message_text(
                f"✅ *Website par publish ho gaya!*\n\n"
                f"📌 {title_preview[:200]}\n\n"
                f"🔗 {PAGE_URLS.get(content_type, 'https://asifmadani.github.io')}",
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
        if content_type == "qa":
            hint = "naya jawab type karke bhejo"
        else:
            hint = "naya title (pehli line) aur description (baaki lines) type karke bhejo"
        await query.edit_message_text(
            f"✏️ *Edit mode:*\n\n{hint.capitalize()}:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_edit|{key}")
            ]]),
        )

    elif action == "discard":
        pending_review.pop(key, None)
        await query.edit_message_text("🗑️ Discard kar diya gaya.")

    elif action == "cancel_edit":
        editing_state.pop(query.from_user.id, None)
        await query.edit_message_text(
            review_text(q),
            parse_mode="Markdown",
            reply_markup=review_keyboard(key),
        )


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Capture edited text from Sheikh — handles both review-edit and delete/edit flows."""
    msg = update.message
    if not msg or msg.chat_id != TG_CHAT_ID:
        return
    user_id = msg.from_user.id if msg.from_user else None
    if not user_id:
        return

    # ── Delete/Edit flow: new content for an existing live item ───────────────
    if user_id in edit_draft:
        draft = edit_draft.pop(user_id)
        lines     = (msg.text or "").strip().split('\n', 1)
        new_title = lines[0].strip()
        new_desc  = lines[1].strip() if len(lines) > 1 else ""

        page      = draft["page"]
        old_block = draft["old_block"]
        new_block = apply_edit(old_block, new_title, new_desc)

        cfg = _PAGE_CFG[page]
        sha, html = await gh_get_file(cfg[0])
        if not sha:
            await msg.reply_text("❌ GitHub fetch fail hua.")
            return
        if old_block not in html:
            await msg.reply_text("❌ Item nahi mila. Shayad already change ho gaya?")
            return

        new_html = html.replace(old_block, new_block, 1)
        ok = await gh_put_file(cfg[0], sha, new_html, f"Edit: {new_title[:50]}")

        if ok:
            await msg.reply_text(
                f"✅ Update ho gaya!\n\n*{new_title}*\n{new_desc[:150]}\n\n"
                f"🔗 {PAGE_URLS.get(page, '')}",
                parse_mode="Markdown",
            )
        else:
            await msg.reply_text("❌ GitHub update fail hua.")
        return

    # ── Review-edit flow: editing pending review draft ─────────────────────────
    if user_id not in editing_state:
        return

    key = editing_state.pop(user_id)
    q   = pending_review.get(key)
    if not q:
        return

    if q.get("type") == "qa":
        q["answer_draft"] = msg.text
    else:
        lines = (msg.text or "").strip().split('\n', 1)
        q["title"] = lines[0].strip()
        if len(lines) > 1:
            q["description"] = lines[1].strip()

    await msg.reply_text(
        review_text(q),
        parse_mode="Markdown",
        reply_markup=review_keyboard(key),
    )


# ══════════════════════════════════════════════════════════════════════════════
# DELETE / EDIT — HTML EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _div_end(html: str, start: int) -> int:
    """Return end position of a <div> block starting at `start`, counting nested divs."""
    depth, i = 0, start
    while i < len(html):
        if html[i:i+4] == "<div":
            depth += 1
            j = html.find(">", i)
            i = j + 1 if j != -1 else i + 4
        elif html[i:i+6] == "</div>":
            depth -= 1
            if depth == 0:
                return i + 6
            i += 6
        else:
            i += 1
    return -1


def find_blocks(html: str, class_name: str, start_after: str = "", stop_before: str = "") -> list[str]:
    """Extract all <div class="class_name">...</div> blocks in a section of html."""
    s, e = 0, len(html)
    if start_after and start_after in html:
        s = html.index(start_after) + len(start_after)
    if stop_before:
        idx = html.find(stop_before, s)
        if idx != -1:
            e = idx

    tag, blocks, pos = f'class="{class_name}"', [], s
    while pos < e:
        ci = html.find(tag, pos, e)
        if ci == -1:
            break
        di = html.rfind("<div", pos, ci)
        if di == -1:
            pos = ci + len(tag)
            continue
        end = _div_end(html, di)
        if end == -1 or end > e:
            break
        blocks.append(html[di:end])
        pos = end
    return blocks


def block_title(block: str) -> str:
    m = re.search(r"<h3>(.*?)</h3>", block, re.DOTALL)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else "Untitled"


def block_desc(block: str) -> str:
    m = re.search(r"<p>(.*?)</p>", block, re.DOTALL)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip()[:200] if m else ""


def apply_edit(old_block: str, new_title: str, new_desc: str) -> str:
    """Replace <h3> and first <p> in a block with new values."""
    b = re.sub(r"<h3>.*?</h3>", f"<h3>{new_title}</h3>", old_block, count=1, flags=re.DOTALL)
    b = re.sub(r"<p>.*?</p>", f"<p>{new_desc}</p>", b, count=1, flags=re.DOTALL)
    return b


# page_type → (filename, div_class, start_after, stop_before)
_PAGE_CFG = {
    "qa":       ("qa.html",       "qa-item",    "<h2>Published Answers</h2>", ""),
    "maqalah":  ("maqalah.html",  "topic-card", 'id="bot-maqalah">',         "<!-- BOT:maqalah -->"),
    "tafseer":  ("tafseer.html",  "topic-card", 'id="bot-tafseer">',         "<!-- BOT:tafseer -->"),
    "tashreeh": ("tashreeh.html", "topic-card", 'id="bot-tashreeh">',        "<!-- BOT:tashreeh -->"),
    "research": ("research.html", "pub-card",   '<div class="pub-list">',    "<!-- BOT:research -->"),
    "books":    ("books.html",    "pub-card",   '<div class="pub-list">',    "<!-- BOT:books -->"),
    "video":    ("videos.html",   "video-card", "<!-- BOT:video -->",         ""),
}

_PAGE_LABEL = {
    "qa": "Q&A", "maqalah": "Maqalah", "tafseer": "Tafseer",
    "tashreeh": "Tashreeh", "research": "Research", "books": "Books", "video": "Videos",
}


async def fetch_blocks(page: str) -> tuple[list[str], str, str] | tuple[None, None, None]:
    cfg = _PAGE_CFG.get(page)
    if not cfg:
        return None, None, None
    filename, cls, start, stop = cfg
    sha, html = await gh_get_file(filename)
    if not sha:
        return None, None, None
    return find_blocks(html, cls, start, stop), sha, html


# ── /delete and /edit command handlers ────────────────────────────────────────

def _page_select_keyboard(op: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("❓ Q&A",       callback_data=f"mgr|{op}|qa"),
            InlineKeyboardButton("📚 Maqalah",   callback_data=f"mgr|{op}|maqalah"),
        ],
        [
            InlineKeyboardButton("📜 Tafseer",   callback_data=f"mgr|{op}|tafseer"),
            InlineKeyboardButton("📋 Tashreeh",  callback_data=f"mgr|{op}|tashreeh"),
        ],
        [
            InlineKeyboardButton("🔬 Research",  callback_data=f"mgr|{op}|research"),
            InlineKeyboardButton("📖 Books",     callback_data=f"mgr|{op}|books"),
        ],
        [InlineKeyboardButton("🎬 Videos",       callback_data=f"mgr|{op}|video")],
        [InlineKeyboardButton("❌ Cancel",        callback_data="mgr|cancel|_")],
    ])


async def on_delete_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or msg.chat_id != TG_CHAT_ID:
        return
    await msg.reply_text(
        "🗑️ *Delete — Kaunsa page?*",
        parse_mode="Markdown",
        reply_markup=_page_select_keyboard("del"),
    )


async def on_edit_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or msg.chat_id != TG_CHAT_ID:
        return
    await msg.reply_text(
        "✏️ *Edit — Kaunsa page?*",
        parse_mode="Markdown",
        reply_markup=_page_select_keyboard("edt"),
    )


async def handle_manage_callback(query, ctx: ContextTypes.DEFAULT_TYPE):
    """Handle all mgr|* callbacks for delete / edit flows."""
    parts   = query.data.split("|")
    op      = parts[1]   # del, edt, item, confirm, cancel
    page    = parts[2] if len(parts) > 2 else "_"
    user_id = query.from_user.id

    if op == "cancel":
        manage_cache.pop(user_id, None)
        edit_draft.pop(user_id, None)
        await query.edit_message_text("❌ Cancel ho gaya.")
        return

    # ── Page selected → show item list ────────────────────────────────────────
    if op in ("del", "edt"):
        await query.edit_message_text("⏳ Items load ho rahe hain...")
        blocks, sha, html = await fetch_blocks(page)
        if blocks is None:
            await query.edit_message_text("❌ Page fetch fail hua.")
            return
        if not blocks:
            await query.edit_message_text(
                f"📭 {_PAGE_LABEL[page]} par abhi koi bot-added item nahi hai."
            )
            return

        manage_cache[user_id] = {"op": op, "page": page, "blocks": blocks}
        emoji = "🗑️" if op == "del" else "✏️"
        text  = f"{emoji} *{_PAGE_LABEL[page]} — item select karo:*\n\n"
        btns  = []
        for i, blk in enumerate(blocks):
            title = block_title(blk)[:40]
            text += f"{i+1}. {title}\n"
            btns.append([InlineKeyboardButton(
                f"{i+1}. {title[:35]}",
                callback_data=f"mgr|item|{page}|{i}",
            )])
        btns.append([InlineKeyboardButton("❌ Cancel", callback_data="mgr|cancel|_")])
        await query.edit_message_text(text, parse_mode="Markdown",
                                      reply_markup=InlineKeyboardMarkup(btns))
        return

    # ── Item selected ──────────────────────────────────────────────────────────
    if op == "item":
        idx   = int(parts[3])
        cache = manage_cache.get(user_id, {})
        if cache.get("page") != page:
            await query.edit_message_text("⚠️ Session expire ho gaya. /delete ya /edit dobara karo.")
            return

        real_op = cache["op"]
        blk     = cache["blocks"][idx]
        title   = block_title(blk)
        desc    = block_desc(blk)

        if real_op == "del":
            await query.edit_message_text(
                f"🗑️ *Delete confirm karo:*\n\n*{title}*\n{desc[:200]}\n\n_Yeh undo nahi hoga!_",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("✅ Haan, Delete!", callback_data=f"mgr|confirm|{page}|{idx}")],
                    [InlineKeyboardButton("❌ Cancel",        callback_data="mgr|cancel|_")],
                ]),
            )
        else:  # edit
            edit_draft[user_id] = {"page": page, "idx": idx, "old_block": blk}
            await query.edit_message_text(
                f"✏️ *Edit — {_PAGE_LABEL[page]}:*\n\n"
                f"*Purana title:* {title}\n"
                f"*Purana content:* {desc[:200]}\n\n"
                "Naya content bhejo:\n"
                "_Line 1 = Title_\n_Line 2+ = Description_",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("❌ Cancel", callback_data="mgr|cancel|_")
                ]]),
            )
        return

    # ── Confirm delete ─────────────────────────────────────────────────────────
    if op == "confirm":
        idx   = int(parts[3])
        cache = manage_cache.pop(user_id, {})
        if not cache or cache.get("page") != page:
            await query.edit_message_text("⚠️ Session expire ho gaya.")
            return

        blk   = cache["blocks"][idx]
        title = block_title(blk)
        cfg   = _PAGE_CFG[page]

        sha, html = await gh_get_file(cfg[0])
        if not sha:
            await query.edit_message_text("❌ GitHub fetch fail hua.")
            return
        if blk not in html:
            await query.edit_message_text("❌ Item HTML mein nahi mila. Shayad already delete ho gaya?")
            return

        new_html = html.replace(blk, "", 1)
        ok = await gh_put_file(cfg[0], sha, new_html, f"Delete: {title[:50]}")

        if ok:
            await query.edit_message_text(
                f"✅ *Delete ho gaya!*\n\n_{title}_\n\n🔗 {PAGE_URLS.get(page, '')}",
                parse_mode="Markdown",
            )
        else:
            await query.edit_message_text("❌ GitHub update fail hua.")


# ══════════════════════════════════════════════════════════════════════════════
# KEEP-ALIVE + FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

async def keep_alive():
    """Ping self every 14 min so Render free tier stays awake."""
    while True:
        await asyncio.sleep(14 * 60)
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                await c.get(WEBHOOK_URL + "/")
            log.info("Keep-alive ping ✓")
        except Exception:
            pass


async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    log.error("PTB error: %s", ctx.error, exc_info=ctx.error)
    try:
        await ctx.bot.send_message(
            chat_id=TG_CHAT_ID,
            text=f"⚠️ Bot internal error:\n{type(ctx.error).__name__}: {ctx.error}",
        )
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    ptb_app.add_handler(CommandHandler("delete", on_delete_cmd))
    ptb_app.add_handler(CommandHandler("edit",   on_edit_cmd))
    ptb_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    ptb_app.add_handler(MessageHandler(
        (filters.TEXT | filters.CAPTION | filters.Document.ALL) & ~filters.COMMAND,
        on_hashtag_msg,
    ))
    ptb_app.add_handler(CallbackQueryHandler(on_callback))
    # group=1 so on_text always runs even after on_hashtag_msg consumed the update in group=0
    ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text), group=1)
    ptb_app.add_error_handler(on_error)

    await ptb_app.initialize()

    # Retry webhook setup (new Render instances may take a moment to get a public DNS)
    for attempt in range(6):
        try:
            wh = f"{WEBHOOK_URL}/webhook/{TG_TOKEN}"
            await ptb_app.bot.set_webhook(wh)
            log.info("Webhook set → %s", wh)
            break
        except Exception as exc:
            log.warning("Webhook attempt %d failed: %s", attempt + 1, exc)
            if attempt < 5:
                await asyncio.sleep(10)
            else:
                raise

    await ptb_app.start()
    asyncio.create_task(keep_alive())
    yield
    await ptb_app.stop()
    await ptb_app.shutdown()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def health():
    return {"status": "ok", "bot": "Darul Ilm CMS Bot ✓"}


@app.post(f"/webhook/{TG_TOKEN}")
async def tg_webhook(request: Request):
    body   = await request.json()
    update = Update.de_json(body, ptb_app.bot)
    await ptb_app.process_update(update)
    return {"ok": True}
