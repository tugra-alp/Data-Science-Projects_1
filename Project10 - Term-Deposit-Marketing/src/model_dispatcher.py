#classification algorithms with best tuned paramters (models/rs_model_selection.py)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

models = {
	"Logistic_Regression": LogisticRegression(C = 100),

	"XGB_Classifier": XGBClassifier(n_estimators= 400, max_depth = 10, learning_rate = 0.1),
    
	"RandomForest_Classifier": RandomForestClassifier(n_jobs=-1, n_estimators = 100, max_depth = 10 ,criterion = 'gini'),
}
