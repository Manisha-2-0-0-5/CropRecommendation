import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
import uuid

class PriceAnalyze:
    def __init__(self, filepath):
        self.prices = pd.read_csv(filepath)
        self._clean_price_data()

    def _clean_price_data(self):
        self.prices.columns = [col.strip().title() for col in self.prices.columns]
        self.prices['Crop Name'] = self.prices['Crop Name'].str.strip().str.title()
        self.prices['Year'] = self.prices['Year'].astype(str)
        self.prices.fillna(0, inplace=True)

    def get_unique_districts(self):
        skip_cols = ['Crop Name', 'Year', 'Season']
        return [col for col in self.prices.columns if col not in skip_cols]

    def get_unique_crops(self):
        return sorted(self.prices['Crop Name'].unique())

    def analyze_prices_for_district(self, crops, district):
        district = district.strip().title()
        if district not in self.prices.columns:
            raise ValueError(f"{district} not found in dataset.")

        filtered = self.prices[self.prices['Crop Name'].isin(crops)][['Crop Name', district]]
        if filtered.empty:
            raise ValueError("No matching data found.")

        filtered[district] = pd.to_numeric(filtered[district], errors='coerce')
        grouped = filtered.groupby('Crop Name')[district].mean().sort_values(ascending=False)

        return grouped

    def plot_prices(self, avg_prices):
        top_crop = avg_prices.idxmax()
        colors = ['crimson' if crop == top_crop else 'skyblue' for crop in avg_prices.index]

        plt.figure(figsize=(10, 6))
        avg_prices.plot(kind='bar', color=colors)
        plt.title('Crop Price Comparison')
        plt.ylabel('Average Price (₹)')
        plt.xlabel('Crop Name')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        filename = f"price_chart_{uuid.uuid4().hex}.png"
        chart_path = os.path.join("static", "charts", filename)
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        return filename

    def recommend_alternatives(self, selected_crops, district):
        district = district.strip().title()
        all_avg_prices = self.prices.groupby('Crop Name')[district].mean().sort_values(ascending=False)

        recommended = all_avg_prices[~all_avg_prices.index.isin(selected_crops)].head(5).reset_index()
        recommended.columns = ['Crop', 'Price']
        return recommended
