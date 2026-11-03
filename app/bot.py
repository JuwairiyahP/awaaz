"""Telegram bot: forward a voice note, get a verdict.

    export TELEGRAM_TOKEN=...            # from @BotFather
    export AWAAZ_CKPT=runs/xlsr/best.pt
    python -m app.bot

Commands
  /start              how to use it
  /enroll <name>      the NEXT voice note you send is saved as <name>'s real voice
  /people             who is enrolled
  /forget <name>      remove someone
Any voice note, audio file or video note is analysed.

Each chat gets its own voice vault (data/vaults/<chat_id>.json), so one
family's enrolled voices are never matched against another's.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from telegram import Update  # noqa: E402
from telegram.constants import ChatAction  # noqa: E402
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters  # noqa: E402

from app.service import get_awaaz, telegram_text  # noqa: E402
from awaaz.speaker import VoiceVault  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("awaaz.bot")

VAULT_DIR = Path(os.environ.get("AWAAZ_VAULT_DIR", "data/vaults"))

HELP = (
    "Namaste! I'm Awaaz.\n\n"
    "Forward me a suspicious voice note and I'll check three things:\n"
    "1. Does the voice sound AI-generated?\n"
    "2. Does it sound like someone you've enrolled?\n"
    "3. Does it follow a known scam script (urgency, secrecy, money, OTP, police)?\n\n"
    "To protect your family, enroll their REAL voices first:\n"
    "/enroll Mum, then send a normal voice note from Mum (20–30 s is ideal).\n\n"
    "/people lists enrolled voices · /forget <name> removes one.\n\n"
    "I don't keep your audio; enrolled voices are stored only as numbers (embeddings)."
)


def vault_for(chat_id: int) -> VoiceVault:
    return VoiceVault(VAULT_DIR / f"{chat_id}.json")


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)


async def enroll_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = " ".join(ctx.args).strip()[:40]
    if not name:
        await update.message.reply_text("Usage: /enroll <name>, e.g. /enroll Mum")
        return
    ctx.chat_data["pending_enroll"] = name
    await update.message.reply_text(f"OK. Now send a voice note of {name}'s REAL voice (20–30 seconds).")


async def people_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    names = vault_for(update.effective_chat.id).names()
    await update.message.reply_text("Enrolled: " + ", ".join(names) if names else "Nobody enrolled yet. Try /enroll Mum")


async def forget_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    name = " ".join(ctx.args).strip()
    v = vault_for(update.effective_chat.id)
    if name not in v.people:
        await update.message.reply_text(f"{name or '(no name)'} isn't enrolled.")
        return
    v.remove(name)
    await update.message.reply_text(f"Removed {name}.")


async def on_audio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    media = msg.voice or msg.audio or msg.video_note or (msg.document if msg.document and
                                                         (msg.document.mime_type or "").startswith(("audio", "video")) else None)
    if media is None:
        return
    await ctx.bot.send_chat_action(msg.chat_id, ChatAction.TYPING)
    tg_file = await media.get_file()
    suffix = Path(tg_file.file_path or "x.ogg").suffix or ".ogg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = Path(tmp.name)
    try:
        await tg_file.download_to_drive(str(path))
        aw = get_awaaz()
        aw.vault = vault_for(msg.chat_id)  # per-chat vault
        pending = ctx.chat_data.pop("pending_enroll", None)
        if pending:
            if aw.encoder is None:
                await msg.reply_text("Speaker matching is switched off on this server.")
                return
            aw.enroll(pending, path)
            await msg.reply_text(f"Saved {pending}'s voice. Now forward any suspicious voice note and I'll compare.")
            return
        result = aw.analyze(path).to_dict()
        await msg.reply_text(telegram_text(result))
        log.info("chat=%s verdict=%s synth=%.2f", msg.chat_id, result["verdict"]["level"], result["synthetic"])
    except Exception as e:  # keep the bot alive and tell the user
        log.exception("analysis failed")
        await msg.reply_text(f"Sorry, I couldn't analyse that clip ({type(e).__name__}). Try a shorter voice note.")
    finally:
        path.unlink(missing_ok=True)


def main():
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise SystemExit("set TELEGRAM_TOKEN (get one from @BotFather)")
    get_awaaz()  # load models before the first message, not during it
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("enroll", enroll_cmd))
    app.add_handler(CommandHandler("people", people_cmd))
    app.add_handler(CommandHandler("forget", forget_cmd))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO | filters.VIDEO_NOTE | filters.Document.ALL, on_audio))
    log.info("Awaaz bot running")
    app.run_polling()


if __name__ == "__main__":
    main()
