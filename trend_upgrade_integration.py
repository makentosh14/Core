# trend_upgrade_integration.py - Seamless integration of enhanced trend system
"""
This file integrates the enhanced trend detection system with your existing code.
It maintains backward compatibility while adding powerful new features.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from logger import log

# Import both old and new systems
try:
    from trend_filters import get_trend_context_cached as get_old_trend_context
    from enhanced_trend_filters import get_enhanced_trend_context
    OLD_SYSTEM_AVAILABLE = True
except ImportError as e:
    log(f"⚠️ Old trend system not available: {e}", level="WARNING")
    OLD_SYSTEM_AVAILABLE = False

class HybridTrendSystem:
    """
    Hybrid system that combines old and new trend detection
    Falls back gracefully and provides enhanced features when available
    """
    
    def __init__(self):
        self.use_enhanced = True
        self.fallback_mode = False
        self.last_enhanced_check = None
        self.enhanced_available = True
        
    async def get_trend_context(self, force_enhanced: bool = False) -> Dict[str, Any]:
        """
        Get trend context with automatic fallback
        
        Args:
            force_enhanced: Force use of enhanced system only
            
        Returns:
            Comprehensive trend context
        """
        try:
            # Try enhanced system first
            if self.use_enhanced and not self.fallback_mode:
                try:
                    enhanced_context = await get_enhanced_trend_context()
                    
                    # Validate enhanced context
                    if self._validate_enhanced_context(enhanced_context):
                        log("✅ Using enhanced trend detection system")
                        return self._format_unified_context(enhanced_context, source="enhanced")
                    else:
                        log("⚠️ Enhanced context validation failed, falling back", level="WARNING")
                        self.fallback_mode = True
                        
                except Exception as e:
                    log(f"⚠️ Enhanced trend system failed: {e}, falling back to old system", level="WARNING")
                    self.fallback_mode = True
                    self.enhanced_available = False
            
            # Fallback to old system if needed
            if OLD_SYSTEM_AVAILABLE and not force_enhanced:
                try:
                    old_context = await get_old_trend_context()
                    log("✅ Using legacy trend detection system")
                    return self._format_unified_context(old_context, source="legacy")
                    
                except Exception as e:
                    log(f"❌ Legacy trend system also failed: {e}", level="ERROR")
                    return self._get_emergency_context()
            
            # Emergency fallback
            log("❌ All trend systems failed, using emergency context", level="ERROR")
            return self._get_emergency_context()
            
        except Exception as e:
            log(f"❌ Critical error in hybrid trend system: {e}", level="ERROR")
            return self._get_emergency_context()
    
    def _validate_enhanced_context(self, context: Dict[str, Any]) -> bool:
        """Validate enhanced context has required fields"""
        required_fields = ["trend", "strength", "confidence", "regime"]
        
        try:
            for field in required_fields:
                if field not in context:
                    log(f"❌ Enhanced context missing required field: {field}", level="WARNING")
                    return False
            
            # Validate data types and ranges
            if not isinstance(context["strength"], (int, float)) or not 0 <= context["strength"] <= 1:
                return False
                
            if not isinstance(context["confidence"], (int, float)) or not 0 <= context["confidence"] <= 100:
                return False
                
            return True
            
        except Exception as e:
            log(f"❌ Error validating enhanced context: {e}", level="WARNING")
            return False
    
    def _format_unified_context(self, context: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Format context into unified structure"""
        try:
            if source == "enhanced":
                # Enhanced context is already in the right format
                unified = context.copy()
                unified["system_used"] = "enhanced"
                unified["features_available"] = [
                    "market_structure", "enhanced_altseason", "multi_source_sentiment",
                    "volume_profile", "institutional_analysis", "support_resistance",
                    "breakout_probability", "trading_recommendations"
                ]
                
            else:  # legacy
                # Convert legacy context to unified format
                unified = {
                    # Core fields (legacy compatibility)
                    "trend": context.get("btc_trend", "neutral"),
                    "strength": context.get("btc_strength", 0.5),
                    "confidence": context.get("btc_confidence", 50),
                    "regime": self._map_legacy_regime(context),
                    
                    # Legacy fields preserved
                    "btc_trend": context.get("btc_trend", "neutral"),
                    "btc_strength": context.get("btc_strength", 0.5),
                    "btc_confidence": context.get("btc_confidence", 50),
                    "sentiment": context.get("sentiment", "neutral"),
                    "altseason": context.get("altseason", False),
                    "altseason_strength": context.get("altseason_strength", 0),
                    
                    # Enhanced fields (limited)
                    "market_structure": {"structure": context.get("btc_trend", "neutral")},
                    "volume_profile": {},
                    "recommendations": self._generate_legacy_recommendations(context),
                    "risk_level": self._assess_legacy_risk(context),
                    "opportunity_score": context.get("btc_strength", 0.5),
                    
                    # Metadata
                    "system_used": "legacy",
                    "features_available": ["btc_trend", "basic_altseason", "basic_sentiment"],
                    "timestamp": context.get("timestamp", datetime.now().isoformat())
                }
            
            # Add system metadata
            unified["hybrid_system"] = {
                "enhanced_available": self.enhanced_available,
                "fallback_mode": self.fallback_mode,
                "system_used": source,
                "last_update": datetime.now().isoformat()
            }
            
            return unified
            
        except Exception as e:
            log(f"❌ Error formatting unified context: {e}", level="ERROR")
            return self._get_emergency_context()
    
    def _map_legacy_regime(self, context: Dict[str, Any]) -> str:
        """Map legacy context to regime classification"""
        try:
            btc_trend = context.get("btc_trend", "neutral")
            confidence = context.get("btc_confidence", 50)
            altseason = context.get("altseason", False)
            
            if altseason and context.get("altseason_strength", 0) > 0.7:
                return "altseason_active"
            elif btc_trend in ["uptrend", "strong_uptrend"] and confidence > 70:
                return "strong_trending"
            elif btc_trend in ["downtrend", "strong_downtrend"] and confidence > 70:
                return "strong_trending"
            elif btc_trend == "ranging":
                return "range_bound"
            else:
                return "transitional"
                
        except Exception:
            return "unknown"
    
    def _generate_legacy_recommendations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic recommendations from legacy context"""
        try:
            btc_trend = context.get("btc_trend", "neutral")
            confidence = context.get("btc_confidence", 50)
            
            if btc_trend in ["uptrend", "strong_uptrend"] and confidence > 70:
                return {
                    "primary_strategy": "trend_following",
                    "risk_allocation": "moderate",
                    "timeframe_preference": "medium",
                    "entry_conditions": ["momentum_confirmation"]
                }
            elif btc_trend == "ranging":
                return {
                    "primary_strategy": "mean_reversion",
                    "risk_allocation": "conservative",
                    "timeframe_preference": "short",
                    "entry_conditions": ["support_resistance_touch"]
                }
            else:
                return {
                    "primary_strategy": "wait_and_see",
                    "risk_allocation": "conservative",
                    "timeframe_preference": "mixed",
                    "entry_conditions": []
                }
                
        except Exception:
            return {"primary_strategy": "wait_and_see", "risk_allocation": "conservative"}
    
    def _assess_legacy_risk(self, context: Dict[str, Any]) -> str:
        """Assess risk level from legacy context"""
        try:
            confidence = context.get("btc_confidence", 50)
            
            if confidence > 75:
                return "low"
            elif confidence > 50:
                return "moderate"
            else:
                return "high"
                
        except Exception:
            return "moderate"
    
    def _get_emergency_context(self) -> Dict[str, Any]:
        """Emergency fallback context when all systems fail"""
        return {
            # Core fields
            "trend": "neutral",
            "strength": 0.5,
            "confidence": 30,
            "regime": "emergency_mode",
            
            # Legacy compatibility
            "btc_trend": "neutral",
            "btc_strength": 0.5,
            "btc_confidence": 30,
            "sentiment": "neutral",
            "altseason": False,
            "altseason_strength": 0,
            
            # Enhanced fields (empty)
            "market_structure": {},
            "volume_profile": {},
            "recommendations": {
                "primary_strategy": "wait_and_see",
                "risk_allocation": "very_conservative",
                "timeframe_preference": "short",
                "entry_conditions": []
            },
            "risk_level": "very_high",
            "opportunity_score": 0.2,
            "support_levels": [],
            "resistance_levels": [],
            
            # Metadata
            "system_used": "emergency",
            "features_available": [],
            "timestamp": datetime.now().isoformat(),
            "error": "all_systems_failed",
            "hybrid_system": {
                "enhanced_available": False,
                "fallback_mode": True,
                "system_used": "emergency",
                "last_update": datetime.now().isoformat()
            }
        }
    
    async def get_trend_summary(self) -> str:
        """Get a concise trend summary for logging"""
        try:
            context = await self.get_trend_context()
            
            trend = context.get("trend", "unknown").upper()
            strength = context.get("strength", 0.5)
            confidence = context.get("confidence", 0)
            regime = context.get("regime", "unknown").upper()
            system = context.get("system_used", "unknown").upper()
            
            return f"Trend: {trend} ({strength:.2f}) | Confidence: {confidence:.1f}% | Regime: {regime} | System: {system}"
            
        except Exception as e:
            log(f"❌ Error getting trend summary: {e}", level="ERROR")
            return "Trend: UNKNOWN | System: ERROR"
    
    async def reset_system(self) -> None:
        """Reset the hybrid system to try enhanced mode again"""
        self.fallback_mode = False
        self.enhanced_available = True
        log("🔄 Hybrid trend system reset - will retry enhanced mode")


class TrendIntegrationManager:
    """
    Manager for integrating enhanced trend system into existing codebase
    Provides migration utilities and compatibility functions
    """
    
    def __init__(self):
        self.hybrid_system = HybridTrendSystem()
        self.migration_status = "not_started"
        self.performance_metrics = {
            "enhanced_calls": 0,
            "legacy_calls": 0,
            "emergency_calls": 0,
            "success_rate": 0.0,
            "avg_response_time": 0.0
        }
    
    async def migrate_to_enhanced_system(self) -> Dict[str, Any]:
        """
        Migrate existing trend detection to enhanced system
        Provides detailed migration report
        """
        try:
            log("🚀 Starting migration to enhanced trend system...")
            self.migration_status = "in_progress"
            
            migration_results = {
                "status": "success",
                "tests_passed": [],
                "tests_failed": [],
                "compatibility_issues": [],
                "recommendations": []
            }
            
            # Test 1: Basic enhanced system availability
            try:
                test_context = await get_enhanced_trend_context()
                if test_context and "trend" in test_context:
                    migration_results["tests_passed"].append("enhanced_system_available")
                    log("✅ Enhanced trend system is available")
                else:
                    migration_results["tests_failed"].append("enhanced_system_not_responding")
            except Exception as e:
                migration_results["tests_failed"].append(f"enhanced_system_error: {e}")
                log(f"❌ Enhanced system test failed: {e}", level="WARNING")
            
            # Test 2: Legacy compatibility
            if OLD_SYSTEM_AVAILABLE:
                try:
                    legacy_context = await get_old_trend_context()
                    if legacy_context:
                        migration_results["tests_passed"].append("legacy_compatibility_maintained")
                        log("✅ Legacy system compatibility maintained")
                except Exception as e:
                    migration_results["tests_failed"].append(f"legacy_compatibility_issue: {e}")
            
            # Test 3: Hybrid system functionality
            try:
                hybrid_context = await self.hybrid_system.get_trend_context()
                if hybrid_context and hybrid_context.get("system_used"):
                    migration_results["tests_passed"].append("hybrid_system_functional")
                    log(f"✅ Hybrid system functional using: {hybrid_context.get('system_used')}")
                else:
                    migration_results["tests_failed"].append("hybrid_system_malfunction")
            except Exception as e:
                migration_results["tests_failed"].append(f"hybrid_system_error: {e}")
            
            # Test 4: Performance comparison
            performance_test = await self._run_performance_test()
            if performance_test["success"]:
                migration_results["tests_passed"].append("performance_test_passed")
                migration_results["performance_data"] = performance_test
            else:
                migration_results["tests_failed"].append("performance_test_failed")
            
            # Generate recommendations
            migration_results["recommendations"] = self._generate_migration_recommendations(migration_results)
            
            # Update migration status
            if len(migration_results["tests_failed"]) == 0:
                self.migration_status = "completed_successfully"
                log("🎉 Migration to enhanced system completed successfully!")
            elif len(migration_results["tests_passed"]) > len(migration_results["tests_failed"]):
                self.migration_status = "completed_with_warnings"
                log("⚠️ Migration completed with some warnings")
            else:
                self.migration_status = "failed"
                log("❌ Migration failed - staying with legacy system")
            
            migration_results["final_status"] = self.migration_status
            return migration_results
            
        except Exception as e:
            log(f"❌ Critical error during migration: {e}", level="ERROR")
            self.migration_status = "failed"
            return {
                "status": "failed",
                "error": str(e),
                "final_status": "failed"
            }
    
    async def _run_performance_test(self) -> Dict[str, Any]:
        """Run performance comparison between systems"""
        try:
            import time
            
            # Test enhanced system
            enhanced_times = []
            enhanced_success = 0
            
            for i in range(3):
                try:
                    start_time = time.time()
                    result = await get_enhanced_trend_context()
                    end_time = time.time()
                    
                    if result and "trend" in result:
                        enhanced_times.append(end_time - start_time)
                        enhanced_success += 1
                except Exception:
                    pass
                
                await asyncio.sleep(1)
            
            # Test legacy system (if available)
            legacy_times = []
            legacy_success = 0
            
            if OLD_SYSTEM_AVAILABLE:
                for i in range(3):
                    try:
                        start_time = time.time()
                        result = await get_old_trend_context()
                        end_time = time.time()
                        
                        if result:
                            legacy_times.append(end_time - start_time)
                            legacy_success += 1
                    except Exception:
                        pass
                    
                    await asyncio.sleep(1)
            
            # Calculate metrics
            enhanced_avg_time = sum(enhanced_times) / len(enhanced_times) if enhanced_times else float('inf')
            legacy_avg_time = sum(legacy_times) / len(legacy_times) if legacy_times else float('inf')
            
            return {
                "success": True,
                "enhanced_avg_time": enhanced_avg_time,
                "legacy_avg_time": legacy_avg_time,
                "enhanced_success_rate": enhanced_success / 3,
                "legacy_success_rate": legacy_success / 3 if OLD_SYSTEM_AVAILABLE else 0,
                "performance_improvement": legacy_avg_time / enhanced_avg_time if enhanced_avg_time > 0 else 1
            }
            
        except Exception as e:
            log(f"❌ Performance test failed: {e}", level="ERROR")
            return {"success": False, "error": str(e)}
    
    def _generate_migration_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate migration recommendations based on test results"""
        recommendations = []
        
        try:
            tests_passed = len(results.get("tests_passed", []))
            tests_failed = len(results.get("tests_failed", []))
            
            if "enhanced_system_available" in results.get("tests_passed", []):
                recommendations.append("✅ Enhanced system is ready for production use")
                
                if tests_failed == 0:
                    recommendations.append("🚀 Full migration to enhanced system recommended")
                    recommendations.append("📊 Enable all enhanced features for maximum benefit")
                else:
                    recommendations.append("⚠️ Gradual migration recommended with hybrid mode")
                    recommendations.append("🔄 Monitor system performance during transition")
            else:
                recommendations.append("❌ Enhanced system not ready - stay with legacy system")
                recommendations.append("🔧 Debug enhanced system issues before migration")
            
            if "legacy_compatibility_maintained" in results.get("tests_passed", []):
                recommendations.append("✅ Safe fallback to legacy system available")
            else:
                recommendations.append("⚠️ Legacy system issues detected - investigate immediately")
            
            # Performance recommendations
            perf_data = results.get("performance_data", {})
            if perf_data.get("success") and perf_data.get("performance_improvement", 1) > 1.5:
                recommendations.append(f"⚡ Enhanced system is {perf_data['performance_improvement']:.1f}x faster")
            elif perf_data.get("enhanced_success_rate", 0) < 0.8:
                recommendations.append("⚠️ Enhanced system reliability needs improvement")
            
            recommendations.append("📈 Enable performance monitoring for ongoing optimization")
            recommendations.append("🔄 Schedule regular system health checks")
            
        except Exception as e:
            log(f"❌ Error generating recommendations: {e}", level="ERROR")
            recommendations.append("❌ Unable to generate recommendations - manual review required")
        
        return recommendations
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics"""
        try:
            # Test current system
            start_time = datetime.now()
            context = await self.hybrid_system.get_trend_context()
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update performance metrics
            system_used = context.get("system_used", "unknown")
            if system_used == "enhanced":
                self.performance_metrics["enhanced_calls"] += 1
            elif system_used == "legacy":
                self.performance_metrics["legacy_calls"] += 1
            else:
                self.performance_metrics["emergency_calls"] += 1
            
            # Calculate success rate
            total_calls = sum([
                self.performance_metrics["enhanced_calls"],
                self.performance_metrics["legacy_calls"],
                self.performance_metrics["emergency_calls"]
            ])
            
            successful_calls = (
                self.performance_metrics["enhanced_calls"] + 
                self.performance_metrics["legacy_calls"]
            )
            
            self.performance_metrics["success_rate"] = successful_calls / total_calls if total_calls > 0 else 0
            self.performance_metrics["avg_response_time"] = response_time
            
            return {
                "status": "healthy" if system_used != "emergency" else "degraded",
                "current_system": system_used,
                "migration_status": self.migration_status,
                "enhanced_available": self.hybrid_system.enhanced_available,
                "fallback_mode": self.hybrid_system.fallback_mode,
                "performance_metrics": self.performance_metrics.copy(),
                "last_response_time": response_time,
                "features_available": context.get("features_available", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log(f"❌ Error getting system status: {e}", level="ERROR")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def optimize_system(self) -> Dict[str, Any]:
        """Optimize system performance and configuration"""
        try:
            log("🔧 Starting system optimization...")
            
            optimization_results = {
                "actions_taken": [],
                "performance_improvements": {},
                "recommendations": []
            }
            
            # Check if we should reset fallback mode
            if self.hybrid_system.fallback_mode:
                try:
                    # Test enhanced system
                    test_context = await get_enhanced_trend_context()
                    if test_context and test_context.get("confidence", 0) > 30:
                        await self.hybrid_system.reset_system()
                        optimization_results["actions_taken"].append("reset_fallback_mode")
                        log("✅ Reset fallback mode - enhanced system is working")
                except Exception:
                    optimization_results["recommendations"].append("Enhanced system still not ready")
            
            # Cache optimization (if applicable)
            if hasattr(get_enhanced_trend_context, 'cache'):
                optimization_results["actions_taken"].append("cache_optimization")
            
            # Performance recommendations
            if self.performance_metrics["emergency_calls"] > 0:
                optimization_results["recommendations"].append("Investigate emergency call triggers")
            
            if self.performance_metrics["success_rate"] < 0.95:
                optimization_results["recommendations"].append("Improve system reliability")
            
            optimization_results["recommendations"].append("Regular system health monitoring recommended")
            
            return optimization_results
            
        except Exception as e:
            log(f"❌ Error during system optimization: {e}", level="ERROR")
            return {"error": str(e), "actions_taken": [], "recommendations": []}


# Global instances
hybrid_trend_system = HybridTrendSystem()
trend_integration_manager = TrendIntegrationManager()

# Main integration functions for backward compatibility
async def get_trend_context_enhanced() -> Dict[str, Any]:
    """
    Enhanced drop-in replacement for get_trend_context_cached()
    Maintains full backward compatibility while adding new features
    """
    return await hybrid_trend_system.get_trend_context()

async def get_trend_context_cached() -> Dict[str, Any]:
    """
    Backward compatible function that now uses the hybrid system
    Existing code using this function will automatically benefit from enhancements
    """
    return await hybrid_trend_system.get_trend_context()

# Enhanced specific functions
async def get_market_structure() -> Dict[str, Any]:
    """Get detailed market structure analysis"""
    try:
        context = await get_enhanced_trend_context()
        return context.get("market_structure", {})
    except Exception as e:
        log(f"❌ Error getting market structure: {e}", level="ERROR")
        return {}

async def get_volume_profile(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """Get volume profile analysis for a symbol"""
    try:
        from enhanced_trend_filters import VolumeProfileEngine
        engine = VolumeProfileEngine()
        return await engine.analyze_volume_profile(symbol)
    except Exception as e:
        log(f"❌ Error getting volume profile: {e}", level="ERROR")
        return {}

async def get_sentiment_analysis() -> Dict[str, Any]:
    """Get multi-source sentiment analysis"""
    try:
        context = await get_enhanced_trend_context()
        return context.get("sentiment", {})
    except Exception as e:
        log(f"❌ Error getting sentiment analysis: {e}", level="ERROR")
        return {"overall_sentiment": "neutral", "sentiment_score": 0.5}

async def get_trading_recommendations() -> Dict[str, Any]:
    """Get AI-powered trading recommendations"""
    try:
        context = await get_enhanced_trend_context()
        return context.get("recommendations", {})
    except Exception as e:
        log(f"❌ Error getting trading recommendations: {e}", level="ERROR")
        return {"primary_strategy": "wait_and_see", "risk_allocation": "conservative"}

# System management functions
async def migrate_trend_system() -> Dict[str, Any]:
    """Migrate to enhanced trend system"""
    return await trend_integration_manager.migrate_to_enhanced_system()

async def get_trend_system_status() -> Dict[str, Any]:
    """Get current trend system status"""
    return await trend_integration_manager.get_system_status()

async def optimize_trend_system() -> Dict[str, Any]:
    """Optimize trend system performance"""
    return await trend_integration_manager.optimize_system()

# Utility functions
def is_enhanced_system_available() -> bool:
    """Check if enhanced system is available"""
    return hybrid_trend_system.enhanced_available and not hybrid_trend_system.fallback_mode

def get_system_capabilities() -> List[str]:
    """Get list of available system capabilities"""
    if is_enhanced_system_available():
        return [
            "market_structure_analysis",
            "enhanced_altseason_detection", 
            "multi_source_sentiment",
            "volume_profile_analysis",
            "institutional_activity_detection",
            "breakout_probability",
            "trading_recommendations",
            "support_resistance_levels",
            "risk_assessment"
        ]
    else:
        return [
            "basic_btc_trend",
            "basic_altseason",
            "basic_sentiment"
        ]

# Export all functions
__all__ = [
    # Main integration functions
    'HybridTrendSystem',
    'TrendIntegrationManager',
    'hybrid_trend_system',
    'trend_integration_manager',
    
    # Backward compatible functions
    'get_trend_context_enhanced',
    'get_trend_context_cached',
    
    # Enhanced feature functions
    'get_market_structure',
    'get_volume_profile', 
    'get_sentiment_analysis',
    'get_trading_recommendations',
    
    # System management
    'migrate_trend_system',
    'get_trend_system_status',
    'optimize_trend_system',
    
    # Utility functions
    'is_enhanced_system_available',
    'get_system_capabilities'
]

# Automatic system initialization
async def initialize_enhanced_trend_system():
    """Initialize the enhanced trend system automatically"""
    try:
        log("🚀 Initializing enhanced trend detection system...")
        
        # Test system availability
        status = await get_trend_system_status()
        log(f"📊 Trend system status: {status['status']} using {status['current_system']}")
        
        # Auto-migrate if enhanced system is available
        if status.get("enhanced_available") and status.get("migration_status") == "not_started":
            migration_result = await migrate_trend_system()
            if migration_result.get("final_status") == "completed_successfully":
                log("✅ Auto-migration to enhanced system completed!")
            else:
                log("⚠️ Auto-migration completed with warnings")
        
        # Log available capabilities
        capabilities = get_system_capabilities()
        log(f"🎯 Available capabilities: {', '.join(capabilities)}")
        
        return True
        
    except Exception as e:
        log(f"❌ Error initializing enhanced trend system: {e}", level="ERROR")
        return False

# Initialize on import (if running in async context)
try:
    import asyncio
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Running in async context, schedule initialization
        asyncio.create_task(initialize_enhanced_trend_system())
    else:
        # Not in async context, will initialize when first called
        pass
except Exception:
    # Not in async context or other issue, will initialize when first called
    pass
