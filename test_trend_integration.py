#!/usr/bin/env python3
"""
Check if your main bot is actually using the enhanced trend system
"""

import asyncio
import sys
import traceback
from datetime import datetime

async def check_bot_integration():
    """Check if the main bot is using enhanced trend system"""
    
    print("🔍 CHECKING BOT INTEGRATION WITH ENHANCED TREND SYSTEM")
    print("=" * 60)
    
    try:
        # Test 1: Check imports
        print("\n📦 Step 1: Testing imports...")
        
        try:
            from trend_upgrade_integration import get_trend_context_cached, hybrid_trend_system
            print("✅ trend_upgrade_integration imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import trend_upgrade_integration: {e}")
            return False
        
        try:
            from enhanced_trend_filters import get_enhanced_trend_context
            print("✅ enhanced_trend_filters imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import enhanced_trend_filters: {e}")
            return False
        
        # Test 2: Check which system is active
        print("\n🎯 Step 2: Checking which trend system is active...")
        
        # Get trend context like your main bot does
        trend_context = await get_trend_context_cached()
        
        system_used = trend_context.get("system_used", "unknown")
        features_available = trend_context.get("features_available", [])
        
        print(f"System Used: {system_used.upper()}")
        print(f"Features Available: {len(features_available)} features")
        
        if system_used == "enhanced":
            print("🎉 SUCCESS! Your bot IS using the enhanced trend system!")
            
            # Show enhanced features being used
            print(f"\n📊 Enhanced Features Active:")
            for feature in features_available[:10]:  # Show first 10
                print(f"   ✅ {feature}")
            if len(features_available) > 10:
                print(f"   ... and {len(features_available) - 10} more features")
                
        elif system_used == "legacy":
            print("⚠️ WARNING: Your bot is using the OLD/LEGACY trend system")
            print("💡 The enhanced system might have failed to initialize")
            
        else:
            print(f"❓ UNKNOWN: System used = {system_used}")
        
        # Test 3: Check trend data quality
        print(f"\n📈 Step 3: Analyzing trend data quality...")
        
        print(f"Trend: {trend_context.get('trend', 'UNKNOWN')}")
        print(f"Strength: {trend_context.get('strength', 'UNKNOWN')}")
        print(f"Confidence: {trend_context.get('confidence', 'UNKNOWN')}%")
        print(f"Regime: {trend_context.get('regime', 'UNKNOWN')}")
        
        # Enhanced-specific fields
        if "recommendations" in trend_context:
            recommendations = trend_context["recommendations"]
            strategy = recommendations.get("primary_strategy", "unknown")
            print(f"🤖 AI Strategy: {strategy.upper()}")
            
        if "opportunity_score" in trend_context:
            opportunity = trend_context["opportunity_score"]
            print(f"🎯 Opportunity Score: {opportunity:.2f}")
            
        if "breakout_probability" in trend_context:
            breakout = trend_context["breakout_probability"]
            if isinstance(breakout, dict):
                prob = breakout.get("probability", "unknown")
                direction = breakout.get("direction", "unknown")
                print(f"💥 Breakout: {prob}% {direction}")
            
        # Test 4: Check system status
        print(f"\n🔧 Step 4: Checking system status...")
        
        try:
            from trend_upgrade_integration import get_trend_system_status
            status = await get_trend_system_status()
            
            print(f"Current System: {status.get('current_system', 'unknown').upper()}")
            print(f"Enhanced Available: {status.get('enhanced_available', False)}")
            print(f"Fallback Mode: {status.get('fallback_mode', 'unknown')}")
            print(f"Success Rate: {status.get('success_rate', 0):.1%}")
            
        except Exception as e:
            print(f"⚠️ Could not get system status: {e}")
        
        # Test 5: Test what your main bot sees
        print(f"\n🤖 Step 5: Testing what your main bot sees...")
        
        # Simulate your main bot's usage
        trend_direction = trend_context.get("trend", "neutral")
        trend_strength = trend_context.get("trend_strength", 0.5)
        
        print(f"Main Bot Sees:")
        print(f"   Trend Direction: {trend_direction}")
        print(f"   Trend Strength: {trend_strength}")
        print(f"   Using System: {system_used}")
        
        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        if system_used == "enhanced":
            print("✅ Your bot is successfully using the enhanced trend system!")
            print("🚀 You're getting advanced market analysis with:")
            print("   • Multi-source sentiment analysis")
            print("   • Enhanced altseason detection")
            print("   • Volume profile analysis")
            print("   • AI-powered trading recommendations")
            print("   • Breakout probability calculations")
            
        else:
            print("⚠️ Your bot is NOT using the enhanced system")
            print("🔧 To fix this, check:")
            print("   1. Enhanced trend system initialization")
            print("   2. Import paths in main.py")
            print("   3. Any errors in enhanced_trend_filters.py")
            
        return system_used == "enhanced"
        
    except Exception as e:
        print(f"❌ Error checking integration: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

async def test_enhanced_features():
    """Test specific enhanced features"""
    
    print("\n🧪 TESTING ENHANCED FEATURES")
    print("=" * 40)
    
    try:
        from trend_upgrade_integration import (
            get_market_structure,
            get_sentiment_analysis,
            get_trading_recommendations
        )
        
        # Test market structure
        print("📊 Testing market structure analysis...")
        structure = await get_market_structure()
        print(f"   Breakout Probability: {structure.get('breakout_probability', {}).get('probability', 'unknown')}")
        
        # Test sentiment
        print("😊 Testing sentiment analysis...")
        sentiment = await get_sentiment_analysis()
        print(f"   Market Mood: {sentiment.get('market_mood', 'unknown')}")
        
        # Test recommendations
        print("🤖 Testing AI recommendations...")
        recommendations = await get_trading_recommendations()
        print(f"   Primary Strategy: {recommendations.get('primary_strategy', 'unknown')}")
        
        print("✅ Enhanced features are working!")
        
    except Exception as e:
        print(f"❌ Enhanced features test failed: {e}")

if __name__ == "__main__":
    async def main():
        integration_working = await check_bot_integration()
        
        if integration_working:
            await test_enhanced_features()
            print("\n🎉 CONCLUSION: Your bot is fully integrated with enhanced trends!")
        else:
            print("\n❌ CONCLUSION: Integration needs attention")
    
    asyncio.run(main())
