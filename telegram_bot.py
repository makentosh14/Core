from config import TELEGRAM_CHAT_ID, TELEGRAM_BOT_TOKEN
import aiohttp
import traceback
import os
import time
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from aiogram.types import InputFile
from logger import LOG_FILE, log
from error_handler import send_error_to_telegram  # ✅ Use only from external file

# === Global message rate limit and flood protection ===
_last_send_time = 0
MIN_MESSAGE_DELAY = 1.5  # seconds
MAX_MESSAGES_PER_MIN = 20
SEND_HISTORY = []

BOT_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot)


async def send_telegram_message(message, retries=3):
    global _last_send_time, SEND_HISTORY
    now = time.time()
    SEND_HISTORY = [t for t in SEND_HISTORY if now - t < 60]  # Keep only last 60 sec

    if len(SEND_HISTORY) >= MAX_MESSAGES_PER_MIN:
        print("⏳ Telegram flood limit hit, delaying message...")
        await asyncio.sleep(5)
        return

    elapsed = now - _last_send_time
    if elapsed < MIN_MESSAGE_DELAY:
        await asyncio.sleep(MIN_MESSAGE_DELAY - elapsed)

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(BOT_URL, data=payload) as resp:
                    _last_send_time = time.time()
                    SEND_HISTORY.append(_last_send_time)
                    return await resp.text()
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2)
                continue
            print(f"❌ Telegram send error: {e}")
            await send_error_to_telegram(f"Telegram Error: {str(e)}")
            return None


def format_trade_signal(
    symbol,
    score,
    tf_scores,
    trend,
    entry_price,
    sl,
    tp1,
    trade_type,
    direction,
    trailing_pct,
    leverage,
    risk_pct,
    confidence=None,
    sl_pct=None,
    tp1_pct=None  # Add this parameter
):
    relevant_tfs = {
        "Scalp": [1, 3],
        "Intraday": [5, 15],
        "Swing": [30, 60, 240]
    }.get(trade_type, [])

    filtered_tf_scores = {k: v for k, v in tf_scores.items() if int(k) in relevant_tfs}
    emoji = "🟢" if direction == "Long" else "🔴"

    # Calculate percentages if not provided
    if sl_pct is None and sl and entry_price:
        sl_pct = abs((sl - entry_price) / entry_price * 100)
    
    if tp1_pct is None and tp1 and entry_price:
        if direction.lower() == "long":
            tp1_pct = (tp1 - entry_price) / entry_price * 100
        else:
            tp1_pct = (entry_price - tp1) / entry_price * 100

    message = (
        f"{emoji} <b>{direction} {trade_type} Signal</b>\n"
        f"<b>Symbol:</b> {symbol}\n"
        f"<b>Score:</b> {score}\n"
        f"<b>TF Breakdown:</b> {filtered_tf_scores}\n\n"
        f"<b>Entry:</b> {entry_price}\n"
        f"<b>SL:</b> {sl}" + (f" ({sl_pct:.2f}%)" if sl_pct is not None else "") + "\n"
        f"<b>TP1:</b> {tp1}" + (f" ({tp1_pct:.2f}%)" if tp1_pct is not None else "") + "\n\n"
        f"⚖️ <b>Risk:</b> {risk_pct:.1f}% of balance\n"
        f"📈 <b>Leverage:</b> {leverage}x\n"
        f"📉 <b>Smart Trailing SL:</b> {trailing_pct:.1f}% after TP1\n"
        f"📊 Trend: BTC = {trend.get('btc_trend', 'N/A')}, Altseason = {trend.get('altseason', 'N/A')}\n"
    )

    if confidence is not None:
        message += f"🔍 <b>Confidence:</b> {confidence:.1f}%"

    return message


async def send_pump_alert(symbol, pump_score, volume_spike_pct, price_change_pct, reason):
    message = (
        f"🚀 <b>Early Pump Signal Detected</b>\n"
        f"<b>Symbol:</b> {symbol}\n"
        f"<b>Pump Score:</b> {pump_score:.2f}\n"
        f"<b>Volume Spike:</b> +{volume_spike_pct:.1f}%\n"
        f"<b>Price Change:</b> +{price_change_pct:.2f}%\n"
        f"<b>Reason:</b> {reason}\n"
        f"⚡ <i>Monitoring for breakout, Smart SL/TP activation on momentum</i>"
    )
    await send_telegram_message(message)


# ✅ Command: /active
@dp.message_handler(commands=["active"])
async def handle_active_trades(message: types.Message):
    from monitor import active_trades  # ⬅️ Lazy import avoids circular import
    active = {k: v for k, v in active_trades.items() if not v.get("exited")}
    if not active:
        await message.reply("📭 No active trades currently being monitored.")
        return

    msg = "📡 <b>Active Trade Setups:</b>\n"
    for symbol, trade in active.items():
        trade_type = trade.get("trade_type", "N/A")
        entry = trade.get("entry_price", "?")
        direction = trade.get("direction", "?")
        trailing_sl = trade.get("trailing_sl", "Not set")
        tp1_hit = "✅" if trade.get("tp1_hit") else "❌"

        msg += (
            f"\n<b>{symbol}</b> | {direction} ({trade_type})\n"
            f"• Entry: {entry}\n"
            f"• Trailing SL: {trailing_sl}\n"
            f"• TP1 Hit: {tp1_hit}\n"
        )

    await message.reply(msg, parse_mode="HTML")


# ✅ Command: /download_trades
@dp.message_handler(commands=["download_trades"])
async def handle_download_trades(message: types.Message):
    if os.path.exists(LOG_FILE):
        file_to_send = InputFile(LOG_FILE)
        await message.reply_document(file_to_send, caption="📁 Trade log file attached.")
    else:
        await message.reply("❌ Log file not found.")


# ✅ New Command: /check_sl [symbol]
@dp.message_handler(commands=["check_sl"])
async def handle_check_sl(message: types.Message):
    args = message.get_args().split()
    symbol = args[0].upper() if args else None
    
    if not symbol:
        await message.reply("⚠️ Please provide a symbol. Example: /check_sl BTCUSDT")
        return
        
    from monitor import active_trades, debug_stop_loss
    
    if symbol not in active_trades:
        await message.reply(f"❌ No active trade found for {symbol}")
        return
        
    # Call the debug function from monitor.py
    await debug_stop_loss(symbol)
    await message.reply(f"✅ SL check for {symbol} completed. Report sent to main channel.")


# ✅ New Command: /restore_sl [symbol]
@dp.message_handler(commands=["restore_sl"])
async def handle_restore_sl(message: types.Message):
    args = message.get_args().split()
    symbol = args[0].upper() if args else None
    
    if not symbol:
        await message.reply("⚠️ Please provide a symbol. Example: /restore_sl BTCUSDT")
        return
        
    from monitor import active_trades, check_and_restore_sl
    
    if symbol not in active_trades:
        await message.reply(f"❌ No active trade found for {symbol}")
        return
        
    await check_and_restore_sl(symbol, active_trades[symbol])
    await message.reply(f"✅ SL restoration for {symbol} initiated.")


# ✅ New Command: /verify_trades
@dp.message_handler(commands=["verify_trades"])
async def handle_verify_trades(message: types.Message):
    from monitor import verify_trade_integrity
    
    await message.reply("🔍 Starting trade verification process...")
    await verify_trade_integrity()
    await message.reply("✅ Trade verification completed. Check main channel for results.")


# ✅ New Command: /update_sl [symbol] [price]
@dp.message_handler(commands=["update_sl"])
async def handle_update_sl(message: types.Message):
    args = message.get_args().split()
    
    if len(args) < 2:
        await message.reply("⚠️ Please provide symbol and price. Example: /update_sl BTCUSDT 25000")
        return
        
    symbol = args[0].upper()
    try:
        price = float(args[1])
    except ValueError:
        await message.reply("❌ Invalid price format. Please provide a valid number.")
        return
        
    from monitor import active_trades, update_stop_loss_order
    
    if symbol not in active_trades:
        await message.reply(f"❌ No active trade found for {symbol}")
        return
        
    trade = active_trades[symbol]
    
    # Validate SL is on the correct side of entry
    direction = trade.get("direction", "").lower()
    entry_price = trade.get("entry_price")
    
    if entry_price:
        if (direction == "long" and price >= entry_price) or (direction == "short" and price <= entry_price):
            await message.reply(f"❌ Invalid SL price! For {direction} positions, SL must be {'below' if direction == 'long' else 'above'} entry price.")
            return
            
    # Validate SL vs current price
    from sl_tp_utils import validate_sl_placement
    validated_price = await validate_sl_placement(symbol, direction, price)
    
    if validated_price != price:
        await message.reply(f"⚠️ SL price adjusted from {price} to {validated_price} to ensure it's on the correct side of market price.")
        price = validated_price
        
    # Update the SL
    result = await update_stop_loss_order(symbol, trade, price)
    
    if result:
        await message.reply(f"✅ SL for {symbol} updated to {price}")
    else:
        await message.reply(f"❌ Failed to update SL for {symbol}. Check logs for details.")


# ✅ New Command: /exit [symbol]
@dp.message_handler(commands=["exit"])
async def handle_exit_trade(message: types.Message):
    args = message.get_args().split()
    symbol = args[0].upper() if args else None
    
    if not symbol:
        await message.reply("⚠️ Please provide a symbol. Example: /exit BTCUSDT")
        return
        
    from monitor import active_trades
    
    if symbol not in active_trades or active_trades[symbol].get("exited"):
        await message.reply(f"❌ No active trade found for {symbol}")
        return
        
    # Execute market exit
    try:
        from bybit_api import place_market_order
        
        trade = active_trades[symbol]
        direction = trade.get("direction", "").lower()
        qty = trade.get("qty")
        
        if not direction or not qty:
            await message.reply(f"❌ Missing trade data for {symbol}")
            return
            
        side = "Sell" if direction == "long" else "Buy"
        
        await message.reply(f"🔄 Executing market exit for {symbol}...")
        
        result = await place_market_order(
            symbol=symbol,
            side=side,
            qty=qty,
            market_type="linear",
            reduce_only=True
        )
        
        if result.get("retCode") == 0:
            # Mark trade as exited
            trade["exited"] = True
            from monitor import save_active_trades
            save_active_trades()
            
            log(f"🚫 Manual exit executed for {symbol} via Telegram command", level="INFO")
            
            # Update trade logs
            from activity_logger import log_trade_to_file
            log_trade_to_file(
                symbol=symbol,
                direction=direction,
                entry=trade.get("entry_price"),
                sl=trade.get("original_sl"),
                tp1=None,
                tp2=None,
                result="manual_exit",
                score=trade.get("score_history", [0])[-1],
                trade_type=trade.get("trade_type", "Unknown"),
                confidence=0
            )
            
            await message.reply(f"✅ Successfully exited {symbol} position.")
            
            # Send to main channel
            await send_telegram_message(f"🚫 <b>Manual Exit</b> for {symbol} via Telegram command")
        else:
            await message.reply(f"❌ Exit failed: {result.get('retMsg')}")
        
    except Exception as e:
        log(f"❌ Error executing manual exit for {symbol}: {e}", level="ERROR")
        await message.reply(f"❌ Error executing exit: {str(e)}")


# ✅ New Command: /status
@dp.message_handler(commands=["status"])
async def handle_bot_status(message: types.Message):
    from monitor import active_trades
    
    active_count = sum(1 for t in active_trades.values() if not t.get("exited"))
    
    # Get latest candle timestamp to verify data freshness
    latest_timestamp = None
    from websocket_candles import live_candles
    
    if live_candles:
        for symbol in live_candles:
            if '1' in live_candles[symbol] and live_candles[symbol]['1']:
                candle = live_candles[symbol]['1'][-1]
                ts = candle.get('timestamp')
                if ts:
                    latest_timestamp = ts
                break
    
    # Format the response
    status_msg = (
        f"🤖 <b>Bot Status Report</b>\n\n"
        f"• Active Trades: {active_count}\n"
    )
    
    if latest_timestamp:
        from datetime import datetime
        dt = datetime.fromtimestamp(latest_timestamp / 1000)  # Convert from ms to seconds
        status_msg += f"• Latest Data: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    # Add uptime if available
    from main import startup_time
    current_time = time.time()
    uptime_seconds = int(current_time - startup_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    status_msg += f"• Uptime: {hours}h {minutes}m {seconds}s\n"
    
    await message.reply(status_msg, parse_mode="HTML")


# ✅ Start the bot
def run_telegram_bot():
    executor.start_polling(dp, skip_updates=True)
