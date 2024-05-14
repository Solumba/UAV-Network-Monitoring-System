from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Establish a connection to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")

# Select the database and collection
db = client["ddos_simulation"]
collection = db["simulation_metrics"]

# Fetch data
data = pd.DataFrame(list(collection.find()))

# Display the first few rows of the dataframe
print(data.head())

data["label"] = (data["latency"] > 50).astype(int)  # 1 for anomalous, 0 for normal

features = data[["latency"]]

# Initialize and fit the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination=0.3)
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


plt.scatter(data["time"], data["latency"], c=data["predicted"], cmap="coolwarm")
plt.title("Anomaly Detection in Network Latency")
plt.xlabel("Time")
plt.ylabel("Latency")
plt.colorbar()
plt.show()
