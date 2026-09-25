"""
Daily Gold Sentiment Bot (Telegram)
Free sources only: Yahoo Finance + Google News RSS + VADER sentiment.
Runs once (GitHub Actions mode) or as a polling bot with /report and /dashboard commands.
"""

import os
import sys
import feedparser
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ------------------ SETTINGS ------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "YOUR_TOKEN_HERE")
CHAT_ID        = int(os.environ.get("CHAT_ID") or "8107598336")
DASHBOARD_URL  = os.environ.get("DASHBOARD_URL", "")   # set via GitHub secret
ROME           = ZoneInfo("Europe/Rome")
# ----------------------------------------------

analyzer = SentimentIntensityAnalyzer()


# ═══════════════════════════════════════════════
#  1. PRICE & TECHNICAL DATA
# ═══════════════════════════════════════════════
def get_price_data():
    df = yf.download("GC=F", period="3mo", interval="1d", progress=False)
    if df.empty:
        raise RuntimeError("No price data returned from Yahoo Finance.")

    def col(name):
        c = df[name]
        return c.iloc[:, 0] if hasattr(c, "iloc") and c.ndim == 2 else c

    close = col("Close")
    high  = col("High")
    low   = col("Low")

    c      = float(close.iloc[-1])
    prev   = float(close.iloc[-2])
    sma20  = float(close.tail(20).mean())
    sma50  = float(close.tail(50).mean())
    hi52   = float(high.tail(252).max())
    lo52   = float(low.tail(252).min())

    day_chg    = (c - prev) / prev * 100
    vs_sma20   = (c - sma20) / sma20 * 100
    vs_sma50   = (c - sma50) / sma50 * 100

    delta  = close.diff().dropna()
    gain   = delta.clip(lower=0).tail(14).mean()
    loss   = (-delta.clip(upper=0)).tail(14).mean()
    rsi    = 100 - 100 / (1 + gain / loss) if loss != 0 else 100.0

    if c > sma20 > sma50:
        trend = ("🟢 Uptrend", "Price is above both SMA-20 and SMA-50")
    elif c < sma20 < sma50:
        trend = ("🔴 Downtrend", "Price is below both SMA-20 and SMA-50")
    else:
        trend = ("🟡 Mixed", "Price is between the two moving averages")

    return dict(
        close=c, prev=prev, day_chg=day_chg,
        day_low=float(low.iloc[-1]), day_high=float(high.iloc[-1]),
        sma20=sma20, sma50=sma50,
        vs_sma20=vs_sma20, vs_sma50=vs_sma50,
        rsi=float(rsi), hi52=hi52, lo52=lo52,
        trend_label=trend[0], trend_desc=trend[1],
    )


# ═══════════════════════════════════════════════
#  2. NEWS SENTIMENT
# ═══════════════════════════════════════════════
def get_news_sentiment():
    feeds = [
        "https://news.google.com/rss/search?q=gold+price&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=XAUUSD&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=gold+market&hl=en-US&gl=US&ceid=US:en",
    ]
    scores, headlines = [], []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            text  = entry.title + ". " + getattr(entry, "summary", "")
            score = analyzer.polarity_scores(text)["compound"]
            scores.append(score)
            headlines.append((score, entry.title, entry.get("link", "")))

    avg = sum(scores) / len(scores) if scores else 0.0
    headlines.sort(key=lambda x: abs(x[0]), reverse=True)
    pos = sum(1 for s in scores if s > 0.05)
    neg = sum(1 for s in scores if s < -0.05)
    neu = len(scores) - pos - neg
    return avg, headlines[:6], pos, neg, neu


# ═══════════════════════════════════════════════
#  3. SCORING HELPERS
# ═══════════════════════════════════════════════
def score_label(score):
    if score >= 65: return "🟢 Bullish"
    if score <= 35: return "🔴 Bearish"
    return "🟡 Neutral"

def rsi_label(rsi):
    if rsi >= 70: return f"{rsi:.0f} ⚠️ Overbought"
    if rsi <= 30: return f"{rsi:.0f} ⚠️ Oversold"
    return f"{rsi:.0f} ✅ Normal"

def bar(score, width=10):
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ═══════════════════════════════════════════════
#  4. BUILD THE TELEGRAM MESSAGE
# ═══════════════════════════════════════════════
def build_message(p, news_avg, headlines, pos, neg, neu):
    tech_score  = max(0, min(100, 50 + p["vs_sma20"] * 4))
    news_score  = (news_avg + 1) / 2 * 100
    overall     = round(0.55 * news_score + 0.45 * tech_score)
    chg_icon    = "🔺" if p["day_chg"] >= 0 else "🔻"
    from52hi    = (p["close"] - p["hi52"]) / p["hi52"] * 100
    from52lo    = (p["close"] - p["lo52"]) / p["lo52"] * 100

    lines = [
        f"🥇 <b>Gold Sentiment Report — {datetime.now(ROME):%d %b %Y, %H:%M}</b>",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📊 <b>OVERALL SCORE: {overall}/100 — {score_label(overall)}</b>",
        f"   [{bar(overall)}]",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "💰 <b>Market Snapshot</b>",
        f"  Price (GC=F):  <b>${p['close']:,.1f}</b>  {chg_icon} {p['day_chg']:+.2f}%",
        f"  Day range:     ${p['day_low']:,.1f} – ${p['day_high']:,.1f}",
        f"  52-wk high:    ${p['hi52']:,.1f}  ({from52hi:+.1f}%)",
        f"  52-wk low:     ${p['lo52']:,.1f}  ({from52lo:+.1f}%)",
        "",
        "📈 <b>Technical Analysis</b>",
        f"  Score: {tech_score:.0f}/100 — {score_label(tech_score)}",
        f"  Trend: {p['trend_label']}",
        f"    └ {p['trend_desc']}",
        f"  SMA-20: ${p['sma20']:,.1f}  ({p['vs_sma20']:+.2f}%)",
        f"  SMA-50: ${p['sma50']:,.1f}  ({p['vs_sma50']:+.2f}%)",
        f"  RSI-14: {rsi_label(p['rsi'])}",
        "",
        "📰 <b>News Sentiment</b>",
        f"  Score: {news_score:.0f}/100 — {score_label(news_score)}",
        f"  Avg VADER compound: {news_avg:+.3f}",
        f"  Articles: 🟢 {pos} bullish · ⚪ {neu} neutral · 🔴 {neg} bearish",
        "",
        "🗞️ <b>Top Headlines</b>",
    ]

    for s, title, link in headlines:
        icon = "🟢" if s > 0.05 else ("🔴" if s < -0.05 else "⚪")
        lines.append(f"  {icon} <a href='{link}'>{title}</a>")

    if DASHBOARD_URL:
        lines += [
            "",
            f"📊 <a href='{DASHBOARD_URL}'>Open Full Dashboard →</a>",
        ]

    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        f"<i>Data: Yahoo Finance · Google News · VADER  |  {datetime.now(ROME):%H:%M %Z}</i>",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════
#  5. SEND REPORT
# ═══════════════════════════════════════════════
async def send_report(bot, chat_id):
    try:
        price = get_price_data()
        news_avg, headlines, pos, neg, neu = get_news_sentiment()
        msg = build_message(price, news_avg, headlines, pos, neg, neu)
    except Exception as e:
        msg = f"⚠️ Error building report:\n<code>{e}</code>"
    await bot.send_message(
        chat_id=chat_id, text=msg,
        parse_mode="HTML", disable_web_page_preview=True,
    )


# ═══════════════════════════════════════════════
#  6. TELEGRAM HANDLERS
# ═══════════════════════════════════════════════
async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Fetching data, one second…")
    await send_report(context.bot, update.effective_chat.id)

async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if DASHBOARD_URL:
        await update.message.reply_text(
            f"📊 <b>Gold Dashboard</b>\n\n"
            f"Open your live dashboard here:\n{DASHBOARD_URL}\n\n"
            f"<i>The page loads live data automatically when you open it.</i>",
            parse_mode="HTML",
            disable_web_page_preview=False,
        )
    else:
        await update.message.reply_text(
            "⚠️ Dashboard URL not configured.\n"
            "Add DASHBOARD_URL to your GitHub secrets and redeploy.",
        )

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dashboard_line = f"\n  /dashboard — open the live web dashboard" if DASHBOARD_URL else ""
    await update.message.reply_text(
        "👋 <b>Gold Sentinel ready!</b>\n\n"
        "Commands:\n"
        "  /report — full gold sentiment report\n"
        f"  /start — show this message{dashboard_line}\n\n"
        "Every weekday at 09:00 (Rome time) I'll send the report automatically.",
        parse_mode="HTML",
    )


# ═══════════════════════════════════════════════
#  7. ENTRY POINT
# ═══════════════════════════════════════════════
def main():
    import asyncio

    if "--once" in sys.argv:
        async def run_once():
            app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            async with app:
                await send_report(app.bot, CHAT_ID)
        asyncio.run(run_once())
        print("Report sent.")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",     start_command))
    app.add_handler(CommandHandler("report",    report_command))
    app.add_handler(CommandHandler("dashboard", dashboard_command))

    async def daily_job(context: ContextTypes.DEFAULT_TYPE):
        now  = datetime.now(ROME)
        sent = context.application.bot_data.setdefault("sent", set())
        if now.hour == 9 and now.minute < 2 and now.date() not in sent:
            sent.add(now.date())
            await send_report(context.bot, CHAT_ID)

    app.job_queue.run_repeating(daily_job, interval=60, first=10)
    print("Bot is running… (Ctrl+C to stop)")
    app.run_polling()


if __name__ == "__main__":
    main()
