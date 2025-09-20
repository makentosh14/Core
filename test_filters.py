import asyncio
import gc
from enhanced_trend_filters import EnhancedTrendOrchestrator

async def main():
    """Test the EnhancedTrendOrchestrator with safe cleanup"""
    orch = None
    try:
        print("🚀 Starting Enhanced Trend Orchestrator test...")
        orch = EnhancedTrendOrchestrator()
        
        # Get trend context
        print("📊 Getting enhanced trend context...")
        ctx = await orch.get_enhanced_trend_context()
        
        # Display results
        print("\n" + "="*50)
        print("ENHANCED TREND ANALYSIS RESULTS")
        print("="*50)
        print(f"Trend: {ctx.get('structure', ctx.get('trend', 'UNKNOWN'))}")
        print(f"Strength: {ctx.get('strength', 'UNKNOWN')}")
        print(f"Breakout: {ctx.get('breakout_probability', ctx.get('breakout', 'UNKNOWN'))}")
        print(f"Key levels: {ctx.get('key_levels', {})}")
        print(f"Confidence: {ctx.get('confidence', 'UNKNOWN')}")
        print(f"Regime: {ctx.get('regime', 'UNKNOWN')}")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Safe cleanup - check if close method exists
        if orch:
            try:
                print("🧹 Attempting cleanup...")
                
                # Try to close if method exists
                if hasattr(orch, 'close') and callable(getattr(orch, 'close')):
                    await orch.close()
                    print("✅ Orchestrator closed successfully")
                else:
                    print("ℹ️  No close method found, attempting manual cleanup...")
                    
                    # Manual cleanup of any sessions in analyzers
                    analyzers = [
                        getattr(orch, 'market_structure', None),
                        getattr(orch, 'altseason_detector', None),
                        getattr(orch, 'sentiment_analyzer', None),
                        getattr(orch, 'volume_engine', None)
                    ]
                    
                    for analyzer in analyzers:
                        if analyzer:
                            # Try to close any session attributes
                            if hasattr(analyzer, 'session') and analyzer.session:
                                if hasattr(analyzer.session, 'close'):
                                    await analyzer.session.close()
                                    print(f"✅ Closed session in {analyzer.__class__.__name__}")
                            
                            # Try to close any api_manager
                            if hasattr(analyzer, 'api_manager') and analyzer.api_manager:
                                if hasattr(analyzer.api_manager, 'close_session'):
                                    await analyzer.api_manager.close_session()
                                    print(f"✅ Closed API manager in {analyzer.__class__.__name__}")
                
                # Force garbage collection to help clean up any remaining references
                gc.collect()
                print("✅ Manual cleanup completed")
                
            except Exception as cleanup_error:
                print(f"⚠️ Error during cleanup: {cleanup_error}")
                # Force garbage collection even if cleanup failed
                gc.collect()

# Helper function to close any remaining aiohttp sessions
async def cleanup_aiohttp_sessions():
    """Force cleanup of any remaining aiohttp sessions"""
    import aiohttp
    import weakref
    
    try:
        # This is a bit hacky, but helps ensure all sessions are closed
        # Get all aiohttp.ClientSession instances from garbage collector
        for obj in gc.get_objects():
            if isinstance(obj, aiohttp.ClientSession):
                if not obj.closed:
                    await obj.close()
                    print("🧹 Closed orphaned aiohttp session")
    except Exception as e:
        print(f"⚠️ Error in session cleanup: {e}")

if __name__ == "__main__":
    try:
        # Run main test
        asyncio.run(main())
        
        print("\n" + "-" * 50)
        print("🧹 Running final cleanup...")
        
        # Run additional cleanup
        asyncio.run(cleanup_aiohttp_sessions())
        
        print("✅ All tests and cleanup completed!")
        
    except KeyboardInterrupt:
        print("\n❌ Test interrupted")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
