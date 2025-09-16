# /root/Core/test_filters.py
import asyncio
from enhanced_trend_filters import EnhancedTrendOrchestrator

async def main():
    orch = EnhancedTrendOrchestrator()
    try:
        ctx = await orch.get_enhanced_trend_context()
        print("Trend:", ctx.get("structure"), "| Strength:", ctx.get("strength"))
        print("Breakout:", ctx.get("breakout_probability"))
        print("Key levels:", ctx.get("key_levels", {}))
    finally:
        # Best-effort: close any underlying HTTP client if exposed
        close = getattr(orch, "aclose", None) or getattr(orch, "close", None)
        if close:
            if asyncio.iscoroutinefunction(close):
                await close()
            else:
                close()

if __name__ == "__main__":
    asyncio.run(main())
