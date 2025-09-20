import asyncio
import gc
import warnings
from enhanced_trend_filters import EnhancedTrendOrchestrator

# Suppress SSL warnings during cleanup
warnings.filterwarnings("ignore", category=ResourceWarning)

async def safe_close_session(session):
    """Safely close an aiohttp session"""
    try:
        if session and not session.closed:
            await session.close()
            # Give the session time to close properly
            await asyncio.sleep(0.1)
            return True
    except Exception as e:
        print(f"⚠️ Error closing session: {e}")
    return False

async def cleanup_analyzer(analyzer, analyzer_name):
    """Safely cleanup an individual analyzer"""
    if not analyzer:
        return
        
    try:
        # Try to close any session attributes
        if hasattr(analyzer, 'session') and analyzer.session:
            if await safe_close_session(analyzer.session):
                print(f"✅ Closed session in {analyzer_name}")
        
        # Try to close any api_manager
        if hasattr(analyzer, 'api_manager') and analyzer.api_manager:
            if hasattr(analyzer.api_manager, 'close_session'):
                await analyzer.api_manager.close_session()
                print(f"✅ Closed API manager in {analyzer_name}")
        
        # Try generic close method
        if hasattr(analyzer, 'close') and callable(getattr(analyzer, 'close')):
            await analyzer.close()
            print(f"✅ Called close() on {analyzer_name}")
            
    except Exception as e:
        print(f"⚠️ Error cleaning up {analyzer_name}: {e}")

async def main():
    """Test the EnhancedTrendOrchestrator with proper cleanup"""
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
        # Cleanup with proper error handling
        if orch:
            try:
                print("🧹 Starting cleanup...")
                
                # Try orchestrator's close method first
                if hasattr(orch, 'close') and callable(getattr(orch, 'close')):
                    await orch.close()
                    print("✅ Orchestrator closed successfully")
                else:
                    print("ℹ️ No close method found, performing manual cleanup...")
                    
                    # Manual cleanup of analyzers
                    await cleanup_analyzer(getattr(orch, 'market_structure', None), 'MarketStructureAnalyzer')
                    await cleanup_analyzer(getattr(orch, 'altseason_detector', None), 'AltseasonDetector')
                    await cleanup_analyzer(getattr(orch, 'sentiment_analyzer', None), 'SentimentAnalyzer')
                    await cleanup_analyzer(getattr(orch, 'volume_engine', None), 'VolumeEngine')
                
                # Clear cache
                if hasattr(orch, 'cache'):
                    orch.cache.clear()
                
                print("✅ Cleanup completed")
                
            except Exception as cleanup_error:
                print(f"⚠️ Error during cleanup: {cleanup_error}")

async def cleanup_remaining_sessions():
    """Clean up any remaining aiohttp sessions before event loop closes"""
    import aiohttp
    
    sessions_found = 0
    sessions_closed = 0
    
    try:
        # Find and close any remaining ClientSession instances
        for obj in gc.get_objects():
            if isinstance(obj, aiohttp.ClientSession):
                sessions_found += 1
                if not obj.closed:
                    if await safe_close_session(obj):
                        sessions_closed += 1
        
        if sessions_found > 0:
            print(f"🧹 Found {sessions_found} sessions, closed {sessions_closed}")
        
        # Give time for connections to close properly
        if sessions_closed > 0:
            await asyncio.sleep(0.2)
            
    except Exception as e:
        print(f"⚠️ Error in final session cleanup: {e}")

async def run_complete_test():
    """Run the complete test with all cleanup"""
    try:
        # Run main test
        await main()
        
        print("\n" + "-" * 50)
        print("🧹 Running final cleanup...")
        
        # Clean up any remaining sessions BEFORE the event loop closes
        await cleanup_remaining_sessions()
        
        # Force garbage collection
        gc.collect()
        
        print("✅ All cleanup completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in test execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        # Use asyncio.run with proper cleanup
        asyncio.run(run_complete_test())
        
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final message
    print("\n🎯 Test execution finished.")
