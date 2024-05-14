# from pymongo import MongoClient
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest
# from sklearn.metrics import classification_report, confusion_matrix

# # Establish a connection to the MongoDB server
# client = MongoClient("mongodb://localhost:27017/")

# # Select the database and collection
# db = client["ddos_simulation"]
# collection = db["simulation_metrics"]

# # Fetch data
# data = pd.DataFrame(list(collection.find()))

# # Display the first few rows of the dataframe
# print(data.head())

# data["label"] = (data["latency"] > 100).astype(int)  # 1 for anomalous, 0 for normal

# features = data[["latency"]]

# # Initialize and fit the Isolation Forest model
# model = IsolationForest(n_estimators=100, contamination=0.2)
# model.fit(features)

# # Predict anomalies (-1 for anomalies, 1 for normal)
# data["predicted"] = model.predict(features)
# data["predicted"] = data["predicted"].apply(
#     lambda x: 1 if x == -1 else 0
# )  # Convert to 0, 1

# print(classification_report(data["label"], data["predicted"]))

# # Display confusion matrix
# cm = confusion_matrix(data["label"], data["predicted"])
# print(cm)


# plt.scatter(data["time"], data["latency"], c=data["predicted"], cmap="coolwarm")
# plt.title("Anomaly Detection in Network Latency")
# plt.xlabel("Time")
# plt.ylabel("Latency")
# plt.colorbar()
# plt.show()

from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Establish a connection to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")

# Select the database and collection
db = client["ddos_simulation"]
collection = db["simulation_metrics"]

# Fetch data
data = pd.DataFrame(list(collection.find()))

# Display the first few rows of the dataframe
print(data.head())

data["label"] = (data["latency"] > 100).astype(int)  # 1 for anomalous, 0 for normal

features = data[["latency"]]

# Initialize and fit the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.2)
model.fit(features)

# Predict anomalies (-1 for anomalies, 1 for normal)
data["predicted"] = model.predict(features)
data["predicted"] = data["predicted"].apply(
    lambda x: 1 if x == -1 else 0
)  # Convert to 0, 1

print(classification_report(data["label"], data["predicted"]))

# Display confusion matrix
cm = confusion_matrix(data["label"], data["predicted"])
print(cm)

from sklearn.preprocessing import MinMaxScaler

# Compute the anomaly scores (the lower, the more anomalous)
scores = model.decision_function(features)

# Since a lower score indicates more of an anomaly, we invert the scores for ROC and PR curve
scores = -scores

# Optionally normalize scores to a [0, 1] range
scaler = MinMaxScaler()
scores_normalized = scaler.fit_transform(scores.reshape(-1, 1)).ravel()

# ROC Curve
fpr, tpr, _ = roc_curve(data["label"], scores_normalized)
roc_auc = auc(fpr, tpr)

plt.scatter(data["time"], data["latency"], c=data["predicted"], cmap="coolwarm")
plt.title("Anomaly Detection in Network Latency")
plt.xlabel("Time")
plt.ylabel("Latency")
plt.colorbar()
plt.show()

plt.hist(scores[data["label"] == 0], bins=50, alpha=0.5, label="Normal")
plt.hist(scores[data["label"] == 1], bins=50, alpha=0.5, label="Anomaly")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.title("Histogram of Anomaly Scores")
plt.show()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(data["label"], scores_normalized)
average_precision = average_precision_score(data["label"], scores_normalized)

plt.subplot(1, 2, 2)
plt.plot(
    recall,
    precision,
    color="blue",
    lw=2,
    label="Precision-Recall curve (area = %0.2f)" % average_precision,
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")

plt.tight_layout()
plt.show()
