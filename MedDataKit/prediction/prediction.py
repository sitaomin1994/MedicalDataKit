from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	mean_squared_error, 
	mean_absolute_error,
	r2_score, 
	accuracy_score, 
	f1_score, 
	roc_auc_score, 
	average_precision_score
)
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
from matplotlib import pyplot as plt

# Prepare data
def train_and_evaluate_model(data, config, model_type = 'nn'):
	
	task_type = config['task_type']
	
	if task_type == 'classification':
	
		# Prepare data
		X = data.drop(config['target_feature'], axis=1)
		y = data[config['target_feature']]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
	
		# Train Neural Network
		if model_type == 'nn':
			nn = MLPClassifier(
				hidden_layer_sizes=(64, 64), random_state=42, max_iter=2000, early_stopping=True, n_iter_no_change=20,
				validation_fraction=0.1, batch_size=256, learning_rate='adaptive', tol=1e-4, learning_rate_init=0.01,
				alpha=0.0
			)
			nn.fit(X_train, y_train)
			y_pred = nn.predict(X_test)
			y_pred_proba = nn.predict_proba(X_test)
			learner = nn
		
		elif model_type == 'xgb':
			
			X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    			X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    		)
			print(X_train_fit.shape, X_val.shape, y_train_fit.shape, y_val.shape)
			
			xgb = XGBClassifier(
				learning_rate=0.03,  # Lower learning rate for better generalization
				max_depth=6,         # Prevent overfitting
				objective='multi:softmax',
				n_estimators=2000,
				min_child_weight=1,
				gamma=0,               # Minimum loss reduction for partition
				subsample=0.8,         # Prevent overfitting
				colsample_bytree=0.8,  # Prevent overfitting
				reg_alpha=0.01,           # L1 regularization
				reg_lambda=0.01,          # L2 regularization
				random_state=42,
				early_stopping_rounds=20,
				eval_metric='mlogloss',
				use_label_encoder=False,
				verbosity=0,
				num_class=len(np.unique(y_train))
			)
			
			xgb.fit(
				X_train_fit, y_train_fit,
				eval_set=[(X_val, y_val)],
				verbose=False
			)
			y_pred = xgb.predict(X_test)
			y_pred_proba = xgb.predict_proba(X_test)
			learner = xgb

		y_test, y_pred_proba = unique_classes_calibration(y_test, y_pred_proba)

		if len(np.unique(y_test)) == 2:
			print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
			print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
			print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
			print(f"PR AUC: {average_precision_score(y_test, y_pred_proba[:, 1]):.4f}")
		else:
			print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
			print(f"F1 Score: {f1_score(y_test, y_pred, average='macro'):.4f}")
			print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba, average='macro', multi_class='ovr'):.4f}")
			print(f"PR AUC: {average_precision_score(y_test, y_pred_proba, average='macro'):.4f}")
	
	else:
		# Prepare data
		X = data.drop(config['target_feature'], axis=1)
		y = data[config['target_feature']]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		# Train Neural Network
		if model_type == 'nn':
			nn = MLPRegressor(
				hidden_layer_sizes=(64, 64), random_state=42, max_iter=2000, early_stopping=True, n_iter_no_change=20,
				validation_fraction=0.1, batch_size=256, learning_rate='adaptive', tol=1e-4, learning_rate_init=0.01,
				alpha=0.0
			)
			nn.fit(X_train, y_train)
			y_pred = nn.predict(X_test)
			learner = nn

		elif model_type == 'xgb':
			X_train_fit, X_val, y_train_fit, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
			print(X_train_fit.shape, X_val.shape, y_train_fit.shape, y_val.shape)

			xgb = XGBRegressor(
				learning_rate=0.03,
				max_depth=6,
				n_estimators=2000,
				min_child_weight=1,
				gamma=0,
				subsample=0.8,
				colsample_bytree=0.8,
				reg_alpha=0.01,
				reg_lambda=0.01,
				random_state=42,
				early_stopping_rounds=20,
				verbosity=0
			)

			xgb.fit(
				X_train_fit, y_train_fit,
				eval_set=[(X_val, y_val)],
				verbose=False
			)
			y_pred = xgb.predict(X_test)
			learner = xgb

		print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
		print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
  
	return learner

def unique_classes_calibration(y_test, y_pred_proba):
	
	if len(np.unique(y_test)) != y_pred_proba.shape[1]:
		present_classes = np.unique(y_test)
		class_mapping = {old_cls: new_idx for new_idx, old_cls in enumerate(present_classes)}
		y_test_remapped = np.array([class_mapping[cls] for cls in y_test])
		y_pred_proba_subset = y_pred_proba[:, present_classes.astype(int)]
		row_sums = y_pred_proba_subset.sum(axis=1, keepdims=True)
		y_pred_proba_normalized = y_pred_proba_subset / row_sums

		return y_test_remapped, y_pred_proba_normalized
	else:
		return y_test, y_pred_proba

def feature_importance(xgb, data, config, k = 20):

	# Get feature importance from XGBoost model
	importance_scores = xgb.feature_importances_
	feature_names = data.drop(config['target_feature'], axis=1).columns

	# Create dataframe of feature importances
	feature_importance = pd.DataFrame({
		'Feature': feature_names,
		'Importance': importance_scores
	})
	feature_importance = feature_importance.sort_values('Importance', ascending=False)

	# Plot feature importances
	plt.figure(figsize=(12, 6))
	plt.bar(range(len(importance_scores)), feature_importance['Importance'])
	plt.xticks(range(len(importance_scores)), feature_importance['Feature'], rotation=90)
	plt.xlabel('Features')
	plt.ylabel('Importance Score')
	plt.title('XGBoost Feature Importance')
	plt.tight_layout()
	plt.show()

	# Print top 10 most important features
	print("\nTop 10 Most Important Features:")
	print(feature_importance.head(k))
 
	return feature_importance
