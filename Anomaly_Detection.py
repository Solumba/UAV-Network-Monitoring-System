from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Establish a connection to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")

# Select the database and collection
db = client["ddos_simulation"]
collection = db["simulation_metrics"]

# Fetch data
data = pd.DataFrame(list(collection.find()))

# Display the first few rows of the dataframe
print(data.head())

# Latency, throughput, and battery
features = data[["latency", "throughput", "battery"]]

# Initialize and fit the Isolation Forest model
model = IsolationForest(n_estimators=100, contamination="auto")
model.fit(features)

# Predict anomalies (-1 for anomalies, 1 for normal)
data["anomaly"] = model.predict(features)

plt.scatter(data.time, data["latency"], c=data["anomaly"], cmap="coolwarm")
plt.title("Anomaly Detection in Network latency")
plt.xlabel("Time")
plt.ylabel("latency")
plt.colorbar()
plt.show()
