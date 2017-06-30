import sklearn.metrics as metrics
y_true = [1, 1, 2, 2, 3, 3]  # Actual, observed testing datset values
y_pred = [1, 1, 1, 3, 2, 3]  # Predicted values from your model

metrics.accuracy_score(y_true, y_pred)
metrics.accuracy_score(y_true, y_pred, normalize=False)

metrics.recall_score(y_true, y_pred, average='weighted')
metrics.recall_score(y_true, y_pred, average=None)

metrics.precision_score(y_true, y_pred, average='weighted')
metrics.precision_score(y_true, y_pred, average=None)

metrics.f1_score(y_true, y_pred, average='weighted')
metrics.f1_score(y_true, y_pred, average=None)

target_names = ['Fruit 1', 'Fruit 2', 'Fruit 3']
metrics.classification_report(y_true, y_pred, target_names=target_names)
