
# ----------------- CRUCIAL TODOS -----------------
# // TODO: prepare training data for parsing and vectorization. need to:
# 1. collect and organize raw data from various sources (e.g. IRL face mocap data, synthetic face mocap data, Quest Pro donations to the dataset)
# 2. clean and preprocess the data (e.g. remove noise, handle missing values, normalize formatting)
# 3. split the data into training, validation, and test sets
# // the prepared data should be in JSONL format, with each line representing a single data point (e.g. a single frame of face mocap data) and containing the following fields:
# example of a data point: {"timestamp_ms": 11.11111111111111, "inputs": [0.6169441938400269, 0.6309791803359985, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0003153681755065918, 0.0003148317337036133, 0.0, 0.0, 0.0, 0.0, 0.00031413629767484963, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "targets": [0.09120509028434753, 0.09120509028434753, 0.07708247750997543, 0.07708247750997543, 0.0, 0.0, 0.0, 0.0], "has_labels": false, "session_id": "synthetic_0d5d4c0108a0"}
# // the "inputs" field should contain the raw input data (e.g. face mocap data) and the "targets" field should contain the corresponding labels (e.g. facial expression labels). the "has_labels" field should indicate whether the data point has labels or not (e.g. synthetic data may not have labels). the "session_id" field should indicate the source of the data point (e.g. IRL face mocap session, synthetic data, Quest Pro donation).
# ---------------- CRUCIAL TODOS -----------------