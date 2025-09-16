import asyncio
from enhanced_trend_filters import EnhancedTrendOrchestrator

async def main():
    orch = EnhancedTrendOrchestrator()
    try:
        ctx = await orch.get_enhanced_trend_context()
        # some builds return "structure", others "trend" – show whichever exists
        trend_label = ctx.get("structure") or ctx.get("trend")
        print("Trend:", trend_label, "| Strength:", ctx.get("strength"))
        print("Breakout:", ctx.get("breakout_probability") or ctx.get("breakout"))
        print("Key levels:", ctx.get("key_levels", {}))
    finally:
        # ensure we close the shared aiohttp client
        close = getattr(orch, "close", None)
        if close:
            await close()

if __name__ == "__main__":
    asyncio.run(main())
