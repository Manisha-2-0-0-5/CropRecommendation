import os

data_path = os.path.join("static", "data", "11.csv")
print(f"Looking for file at: {data_path}")  # Debugging line

if not os.path.exists(data_path):
    raise FileNotFoundError(f"The file {data_path} does not exist.")

analyzer = PriceAnalyze(data_path)
