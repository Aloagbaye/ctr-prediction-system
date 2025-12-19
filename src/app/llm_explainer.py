"""
LLM-based explanation generator for CTR predictions

Uses language models to generate human-readable explanations
"""

import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LLMExplainer:
    """Generate explanations for CTR predictions using LLM"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize LLM explainer
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.use_llm = self.api_key is not None
        
        if not self.use_llm:
            logger.warning("No OpenAI API key found. Using rule-based explanations.")
    
    def explain_prediction(
        self,
        prediction: Dict[str, Any],
        request_data: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate explanation for a prediction
        
        Args:
            prediction: Prediction result dictionary
            request_data: Original request data
            feature_importance: Optional feature importance scores
            
        Returns:
            Human-readable explanation string
        """
        if self.use_llm:
            return self._llm_explain(prediction, request_data, feature_importance)
        else:
            return self._rule_based_explain(prediction, request_data, feature_importance)
    
    def _llm_explain(
        self,
        prediction: Dict[str, Any],
        request_data: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> str:
        """Generate explanation using LLM"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            # Build prompt
            prompt = self._build_prompt(prediction, request_data, feature_importance)
            
            # Call LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data scientist explaining CTR (Click-Through Rate) predictions. Provide clear, concise explanations that help users understand why an ad impression received a particular CTR prediction."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except ImportError:
            logger.error("OpenAI library not installed. Falling back to rule-based explanation.")
            return self._rule_based_explain(prediction, request_data, feature_importance)
        except Exception as e:
            logger.error(f"LLM explanation failed: {e}. Falling back to rule-based explanation.")
            return self._rule_based_explain(prediction, request_data, feature_importance)
    
    def _build_prompt(
        self,
        prediction: Dict[str, Any],
        request_data: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> str:
        """Build prompt for LLM"""
        ctr = prediction.get('predicted_ctr', 0)
        model_name = prediction.get('model_name', 'unknown')
        
        prompt = f"""Explain why this ad impression received a CTR prediction of {ctr:.2%}.

Context:
- User ID: {request_data.get('user_id', 'N/A')}
- Ad ID: {request_data.get('ad_id', 'N/A')}
- Device: {request_data.get('device', 'N/A')}
- Placement: {request_data.get('placement', 'N/A')}
- Hour: {request_data.get('hour', 'N/A')}
- Day of Week: {request_data.get('day_of_week', 'N/A')}
- Model Used: {model_name}
"""
        
        if feature_importance:
            prompt += "\nTop contributing factors:\n"
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for feature, importance in sorted_features:
                prompt += f"- {feature}: {importance:.3f}\n"
        
        prompt += "\nProvide a brief, user-friendly explanation (2-3 sentences) explaining the prediction."
        
        return prompt
    
    def _rule_based_explain(
        self,
        prediction: Dict[str, Any],
        request_data: Dict[str, Any],
        feature_importance: Optional[Dict[str, float]] = None
    ) -> str:
        """Generate rule-based explanation without LLM"""
        ctr = prediction.get('predicted_ctr', 0)
        device = request_data.get('device', 'unknown')
        placement = request_data.get('placement', 'unknown')
        hour = request_data.get('hour')
        
        # Base explanation
        explanation = f"This ad impression has a predicted CTR of **{ctr:.2%}**."
        
        # Add context-based insights
        insights = []
        
        # Device insights
        if device == 'mobile':
            insights.append("Mobile devices typically show higher engagement.")
        elif device == 'desktop':
            insights.append("Desktop users may be more selective.")
        
        # Placement insights
        if placement == 'header':
            insights.append("Header placements often receive more visibility.")
        elif placement == 'sidebar':
            insights.append("Sidebar placements may have lower visibility.")
        elif placement == 'popup':
            insights.append("Popup ads can have higher engagement but also higher annoyance.")
        
        # Time-based insights
        if hour is not None:
            if 9 <= hour <= 17:
                insights.append("Business hours typically show higher engagement.")
            elif 18 <= hour <= 22:
                insights.append("Evening hours may have higher user activity.")
            else:
                insights.append("Off-peak hours may show lower engagement.")
        
        # CTR interpretation
        if ctr < 0.01:
            insights.append("This is a relatively low CTR, suggesting limited user interest.")
        elif ctr < 0.05:
            insights.append("This is a moderate CTR, indicating reasonable user interest.")
        else:
            insights.append("This is a high CTR, suggesting strong user interest.")
        
        # Combine explanation
        if insights:
            explanation += " " + " ".join(insights[:3])  # Limit to 3 insights
        
        # Add feature importance if available
        if feature_importance:
            top_feature = max(feature_importance.items(), key=lambda x: abs(x[1]))
            explanation += f" The most important factor is **{top_feature[0]}** (importance: {top_feature[1]:.3f})."
        
        return explanation
    
    def explain_comparison(
        self,
        predictions: Dict[str, Dict[str, Any]],
        request_data: Dict[str, Any]
    ) -> str:
        """
        Explain comparison between multiple models
        
        Args:
            predictions: Dictionary mapping model names to predictions
            request_data: Original request data
            
        Returns:
            Comparison explanation
        """
        if len(predictions) < 2:
            return "Need at least 2 models for comparison."
        
        # Find best and worst predictions
        sorted_preds = sorted(
            predictions.items(),
            key=lambda x: x[1].get('predicted_ctr', 0),
            reverse=True
        )
        
        best_model, best_pred = sorted_preds[0]
        worst_model, worst_pred = sorted_preds[-1]
        
        best_ctr = best_pred.get('predicted_ctr', 0)
        worst_ctr = worst_pred.get('predicted_ctr', 0)
        difference = best_ctr - worst_ctr
        
        explanation = f"""
**Model Comparison:**

- **Best Prediction**: {best_model.upper()} with {best_ctr:.2%} CTR
- **Lowest Prediction**: {worst_model.upper()} with {worst_ctr:.2%} CTR
- **Difference**: {difference:.2%} ({difference/best_ctr*100:.1f}% relative difference)

The models show {"significant" if difference > 0.01 else "moderate"} variation in their predictions. 
{best_model.upper()} is more {"optimistic" if best_ctr > 0.02 else "conservative"} about this impression's potential.
"""
        
        return explanation.strip()

