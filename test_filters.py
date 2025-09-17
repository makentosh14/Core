import asyncio
from enhanced_trend_filters import EnhancedTrendOrchestrator

async def main():
    orch = EnhancedTrendOrchestrator()
    try:
        ctx = await orch.get_enhanced_trend_context()
        print("Trend:", ctx.get("structure") or ctx.get("trend"), "| Strength:", ctx.get("strength"))
        print("Breakout:", ctx.get("breakout_probability") or ctx.get("breakout"))
        print("Key levels:", ctx.get("key_levels", {}))
    finally:
        await orch.close()

if __name__ == "__main__":
    asyncio.run(main())
