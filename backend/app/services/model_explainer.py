import shap
import numpy as np
from typing import Dict, List, Any

class ModelExplainer:
    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        self.explainer = None
    
    def _create_explainer(self, X_background: np.ndarray):
        if self.model_type in ["xgboost", "lightgbm"]:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            background = shap.sample(X_background, min(100, len(X_background)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)
    
    def explain(self, X: np.ndarray, feature_names: List[str], max_samples: int = 100) -> Dict[str, Any]:
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_explain = X[indices]
        else:
            X_explain = X
        
        if self.explainer is None:
            self._create_explainer(X_explain)
        
        shap_values = self.explainer.shap_values(X_explain)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        feature_importance = {}
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        for i, name in enumerate(feature_names):
            feature_importance[name] = float(mean_abs_shap[i])
        
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "feature_importance": feature_importance,
            "feature_names": feature_names,
            "summary": {
                "num_samples_explained": len(X_explain),
                "num_features": len(feature_names),
                "top_features": list(feature_importance.keys())[:5]
            }
        }

def get_shap_explanation(model, model_type: str, X_train: np.ndarray, X_explain: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    explainer = ModelExplainer(model, model_type)
    explainer._create_explainer(X_train)
    return explainer.explain(X_explain, feature_names)
