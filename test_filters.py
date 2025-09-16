python3 - <<'PY'
import asyncio
from enhanced_trend_filters import EnhancedTrendOrchestrator

async def main():
    orch = EnhancedTrendOrchestrator()
    ctx = await orch.get_enhanced_trend_context()
    print("Trend:", ctx.get("structure"), "| Strength:", ctx.get("strength"))
    print("Breakout:", ctx.get("breakout_probability"))
    print("Key levels:", ctx.get("key_levels", {}))

asyncio.run(main())
PY
