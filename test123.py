import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class TamilNaduAgroExpertML:
    def __init__(self):
        try:
            # Load datasets
            self.crops = pd.read_csv(r"C:\Users\manis\Downloads\top15_tamilnadu_crops.csv")
            self.rainfall = pd.read_csv(r"C:\Users\manis\Downloads\final_rainfall_2021_25.csv")
            self.nutrients = pd.read_csv(r"C:\Users\manis\Downloads\crop_nutrients.csv")

            # Clean and prepare the data
            self._clean_data()

            # Create dataset
            self.dataset = self._create_training_dataset()
            if 'District' not in self.dataset.columns or 'Seasons' not in self.dataset.columns:
                raise ValueError("Dataset missing required columns after preparation.")

            # Encode categorical variables
            self.le_district = LabelEncoder()
            self.le_crop = LabelEncoder()
            self.le_season = LabelEncoder()

            self.dataset['District_encoded'] = self.le_district.fit_transform(self.dataset['District'])
            self.dataset['Season_encoded'] = self.le_season.fit_transform(self.dataset['Seasons'])
            self.dataset['Crop_encoded'] = self.le_crop.fit_transform(self.dataset['Crop'])

            # Define features and target
            features = ['District_encoded', 'Season_encoded', 'Duration', 'Irrigation', 'Rainfall', 'N', 'P', 'K']
            target = 'Crop_encoded'

            X = self.dataset[features]
            y = self.dataset[target]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)

            print("✅ Model trained successfully.")

        except Exception as e:
            print(f"❌ Error during initialization: {e}")

    def _clean_data(self):
        """Normalize and clean column names and values"""
        # Normalize column names
        self.crops.columns = self.crops.columns.str.strip().str.lower()
        self.rainfall.columns = self.rainfall.columns.str.strip().str.lower()
        self.nutrients.columns = self.nutrients.columns.str.strip().str.lower()

        # Standardize string values
        self.rainfall['district'] = self.rainfall['district'].str.strip().str.title()
        self.nutrients['district'] = self.nutrients['district'].str.strip().str.title()

        # Handle missing or malformed values
        self.crops['duration'] = pd.to_numeric(self.crops['duration'], errors='coerce').fillna(90)

        # Calculate total rainfall for each district
        self.rainfall['rainfall'] = (
            self.rainfall['actual rainfall in south west monsoon (june\'yyyy to september\'yyyy) in mm'].fillna(0) +
            self.rainfall['actual rainfall in north east monsoon (october\'yyyy to december\'yyyy) in mm'].fillna(0) +
            self.rainfall['actual rainfall in winter season (january\'yyyy to february\'yyyy) in mm'].fillna(0) +
            self.rainfall['actual rainfall in hot weather season (march\'yyyy to may\'yyyy) in mm'].fillna(0)
        )

    def _create_training_dataset(self):
        """Build a training dataset of all district-season-crop combinations"""
        all_combinations = []
        for _, rain_row in self.rainfall.iterrows():
            district = rain_row['district']
            rainfall = rain_row['rainfall']

            nutrient_row = self.nutrients[self.nutrients['district'] == district]
            if nutrient_row.empty:
                continue  # Skip districts with missing nutrient info

            N = nutrient_row['n_high_%'].values[0]
            P = nutrient_row['p_high_%'].values[0]
            K = nutrient_row['k_high_%'].values[0]

            for _, crop_row in self.crops.iterrows():
                crop = crop_row['crop']
                duration = crop_row['duration']
                min_rain = crop_row['min_rain']
                max_rain = crop_row['max_rain']
                min_N = crop_row['min_n']
                min_P = crop_row['min_p']
                min_K = crop_row['min_k']

                score = 1 if (min_rain <= rainfall <= max_rain and N >= min_N and P >= min_P and K >= min_K) else 0

                for season in ['Kharif', 'Rabi', 'Summer', 'Winter']:
                    all_combinations.append({
                        'District': district,
                        'Seasons': season,
                        'Crop': crop,
                        'Duration': duration,
                        'Irrigation': 1,  # You can customize this based on input
                        'Rainfall': rainfall,
                        'N': N,
                        'P': P,
                        'K': K,
                        'Crop_Suitable': score
                    })

        return pd.DataFrame(all_combinations)

    def _season_to_int(self, season):
        season = season.title()
        if season not in self.le_season.classes_:
            return -1
        return self.le_season.transform([season])[0]

    def _encode_district(self, district):
        district = district.title()
        if district not in self.le_district.classes_:
            return -1
        return self.le_district.transform([district])[0]

    def recommend(self, district, month, irrigation, duration):
        try:
            season = self._get_season(month)
            district = district.title()

            season_encoded = self._season_to_int(season)
            district_encoded = self._encode_district(district)

            if season_encoded == -1 or district_encoded == -1:
                return "Selected district or season not recognized by the system."

            rainfall_row = self.rainfall[(self.rainfall['district'] == district)]
            nutrient_row = self.nutrients[self.nutrients['district'] == district]

            if rainfall_row.empty or nutrient_row.empty:
                return "No data available for the selected district and season."

            rainfall = rainfall_row['rainfall'].values[0]
            N = nutrient_row['n_high_%'].values[0]
            P = nutrient_row['p_high_%'].values[0]
            K = nutrient_row['k_high_%'].values[0]

            input_features = [[district_encoded, season_encoded, duration, irrigation, rainfall, N, P, K]]
            crop_encoded = self.model.predict(input_features)[0]
            recommended_crop = self.le_crop.inverse_transform([crop_encoded])[0]

            return recommended_crop
        except Exception as e:
            return f"Error in recommendation: {e}"

    def _get_season(self, month):
        season_map = {
            'january': 'Winter', 'february': 'Winter', 'march': 'Summer', 'april': 'Summer',
            'may': 'Summer', 'june': 'Kharif', 'july': 'Kharif', 'august': 'Kharif',
            'september': 'Kharif', 'october': 'Rabi', 'november': 'Rabi', 'december': 'Rabi'
        }
        return season_map.get(month.lower(), 'Unknown')


# Example usage
if __name__ == "__main__":
    expert = TamilNaduAgroExpertML()

    district = "Salem"
    month = "August"
    irrigation = 1
    duration = 110

    crop = expert.recommend(district, month, irrigation, duration)
    print(f"🌾 Recommended crop for {district} in {month}: {crop}")
