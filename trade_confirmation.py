"""
Trade confirmation logic for pattern-indicator alignment
"""

from pattern_detector import get_pattern_direction, analyze_pattern_strength, REVERSAL_PATTERNS, CONTINUATION_PATTERNS
from logger import log

class TradeConfirmation:
    def __init__(self):
        self.confirmation_rules = {
            "Scalp": {
                "min_pattern_strength": 0.6,
                "required_confirmations": 2,
                "max_opposing_signals": 1,
                "primary_indicators": ["macd", "ema", "volume_spike", "whale"],
                "pattern_bonus": 1.5,
                "timeframes": ["1", "3"]
            },
            "Intraday": {
                "min_pattern_strength": 0.7,
                "required_confirmations": 3,
                "max_opposing_signals": 0,
                "primary_indicators": ["supertrend", "ema_ribbon", "macd", "volume"],
                "pattern_bonus": 1.2,
                "timeframes": ["5", "15"]
            },
            "Swing": {
                "min_pattern_strength": 0.8,
                "required_confirmations": 3,
                "max_opposing_signals": 0,
                "primary_indicators": ["rsi", "supertrend", "ema_ribbon", "bollinger"],
                "pattern_bonus": 1.0,
                "timeframes": ["30", "60", "240"]
            }
        }
        
        # Pattern preferences by trade type
        self.pattern_preferences = {
            "Scalp": {
                "preferred": ["hammer", "marubozu", "bullish_engulfing", "bearish_engulfing", 
                             "bullish_kicker", "bearish_kicker"],
                "avoid": ["doji", "spinning_top", "harami"]
            },
            "Intraday": {
                "preferred": ["morning_star", "evening_star", "three_white_soldiers", 
                             "three_black_crows", "piercing_line", "dark_cloud_cover"],
                "avoid": ["doji"]
            },
            "Swing": {
                "preferred": ["morning_star", "evening_star", "bullish_abandoned_baby", 
                             "bearish_abandoned_baby", "three_white_soldiers", "three_black_crows"],
                "avoid": []
            }
        }
    
    def validate_entry(self, symbol, trade_type, pattern, indicator_scores, direction, candles_by_tf):
        """
        Validate if all conditions are met for trade entry
        
        Returns: (is_valid, reason, confidence_adjustment)
        """
        rules = self.confirmation_rules[trade_type]
        validation_results = []
        
        # 1. Check pattern quality
        pattern_check = self._validate_pattern(pattern, trade_type, direction, candles_by_tf)
        validation_results.append(pattern_check)
        
        if not pattern_check[0] and pattern is not None:
            return False, pattern_check[1], 0
        
        # 2. Check indicator confirmations
        indicator_check = self._validate_indicators(indicator_scores, direction, trade_type)
        validation_results.append(indicator_check)
        
        if not indicator_check[0]:
            return False, indicator_check[1], 0
        
        # 3. Check for conflicting signals
        conflict_check = self._check_conflicts(indicator_scores, direction, rules)
        validation_results.append(conflict_check)
        
        if not conflict_check[0]:
            return False, conflict_check[1], 0
        
        # 4. Calculate confidence adjustment based on alignment
        confidence_adj = self._calculate_confidence_adjustment(
            pattern, indicator_scores, direction, trade_type
        )
        
        # All checks passed
        return True, "All confirmations met", confidence_adj
    
    def _validate_pattern(self, pattern, trade_type, direction, candles_by_tf):
        """Validate pattern quality and direction alignment"""
        if pattern is None:
            # Pattern not required but if missing, need more indicator confirmations
            return True, "No pattern detected - requiring extra confirmations", -0.1
        
        # Check if pattern is preferred for this trade type
        prefs = self.pattern_preferences[trade_type]
        if pattern in prefs["avoid"]:
            return False, f"Pattern {pattern} not suitable for {trade_type} trades", 0
        
        # Get pattern direction
        pattern_dir = get_pattern_direction(pattern)
        trade_dir = "bullish" if direction == "Long" else "bearish"
        
        # Check direction alignment
        if pattern_dir == "neutral":
            return True, "Neutral pattern detected", 0
        elif pattern_dir != trade_dir:
            return False, f"Pattern direction ({pattern_dir}) conflicts with trade direction ({trade_dir})", 0
        
        # Check pattern strength across relevant timeframes
        rules = self.confirmation_rules[trade_type]
        min_strength = rules["min_pattern_strength"]
        
        # Check pattern on multiple timeframes for confirmation
        confirmed_tfs = 0
        total_strength = 0
        
        for tf in rules["timeframes"]:
            if tf in candles_by_tf:
                candles = candles_by_tf[tf]
                if len(candles) >= 3:
                    detected = detect_pattern(candles)
                    if detected == pattern:
                        strength = analyze_pattern_strength(pattern, candles)
                        total_strength += strength
                        confirmed_tfs += 1
        
        if confirmed_tfs == 0:
            return False, f"Pattern {pattern} not confirmed on any {trade_type} timeframe", 0
        
        avg_strength = total_strength / confirmed_tfs
        if avg_strength < min_strength:
            return False, f"Pattern strength too low: {avg_strength:.2f} < {min_strength}", 0
        
        # Bonus for preferred patterns
        bonus = 0.2 if pattern in prefs["preferred"] else 0
        
        return True, f"Pattern {pattern} confirmed with strength {avg_strength:.2f}", bonus
    
    def _validate_indicators(self, indicator_scores, direction, trade_type):
        """Validate indicator confirmations"""
        rules = self.confirmation_rules[trade_type]
        primary_indicators = rules["primary_indicators"]
        
        # Count confirmations from primary indicators
        confirmations = 0
        confirming_indicators = []
        
        for indicator_key, score in indicator_scores.items():
            # Check if this is a primary indicator
            for primary in primary_indicators:
                if primary in indicator_key.lower():
                    if direction == "Long" and score > 0:
                        confirmations += 1
                        confirming_indicators.append(f"{indicator_key}({score:.2f})")
                        break
                    elif direction == "Short" and score < 0:
                        confirmations += 1
                        confirming_indicators.append(f"{indicator_key}({score:.2f})")
                        break
        
        if confirmations < rules["required_confirmations"]:
            return False, f"Insufficient confirmations: {confirmations} < {rules['required_confirmations']} " \
                         f"(found: {', '.join(confirming_indicators)})", 0
        
        return True, f"Confirmed by: {', '.join(confirming_indicators)}", 0
    
    def _check_conflicts(self, indicator_scores, direction, rules):
        """Check for conflicting signals"""
        opposing_signals = 0
        opposing_indicators = []
        
        for indicator_key, score in indicator_scores.items():
            # Strong opposing signal threshold
            if direction == "Long" and score < -0.5:
                opposing_signals += 1
                opposing_indicators.append(f"{indicator_key}({score:.2f})")
            elif direction == "Short" and score > 0.5:
                opposing_signals += 1
                opposing_indicators.append(f"{indicator_key}({score:.2f})")
        
        if opposing_signals > rules["max_opposing_signals"]:
            return False, f"Too many opposing signals ({opposing_signals}): {', '.join(opposing_indicators)}", 0
        
        return True, "No significant conflicts", 0
    
    def _calculate_confidence_adjustment(self, pattern, indicator_scores, direction, trade_type):
        """Calculate confidence adjustment based on signal alignment"""
        adjustment = 0
        rules = self.confirmation_rules[trade_type]
        
        # Pattern bonus
        if pattern and pattern in self.pattern_preferences[trade_type]["preferred"]:
            adjustment += 0.1
        
        # Count aligned indicators
        aligned_count = 0
        total_count = 0
        
        for indicator_key, score in indicator_scores.items():
            if abs(score) > 0.1:  # Significant signal
                total_count += 1
                if (direction == "Long" and score > 0) or (direction == "Short" and score < 0):
                    aligned_count += 1
        
        # Calculate alignment ratio
        if total_count > 0:
            alignment_ratio = aligned_count / total_count
            if alignment_ratio > 0.8:
                adjustment += 0.15
            elif alignment_ratio > 0.6:
                adjustment += 0.05
            else:
                adjustment -= 0.1
        
        return adjustment
    
    def get_pattern_score_multiplier(self, pattern, trade_type):
        """Get the score multiplier for a pattern based on trade type"""
        if pattern is None:
            return 1.0
        
        rules = self.confirmation_rules[trade_type]
        base_multiplier = rules.get("pattern_bonus", 1.0)
        
        # Additional multiplier for preferred patterns
        if pattern in self.pattern_preferences[trade_type]["preferred"]:
            return base_multiplier * 1.2
        elif pattern in self.pattern_preferences[trade_type]["avoid"]:
            return 0.5  # Reduce score for avoided patterns
        else:
            return base_multiplier

# Instantiate for use
trade_confirmation = TradeConfirmation()
