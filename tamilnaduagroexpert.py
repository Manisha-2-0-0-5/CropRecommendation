import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

class TamilNaduAgroExpert:
    def __init__(self):
        try:
            # Initialize label encoders first
            self.le_district = LabelEncoder()
            self.le_crop = LabelEncoder()
            self.le_season = LabelEncoder()
            self.le_duration = LabelEncoder()

            # Load datasets
            self.crops = pd.read_csv("top15_tamilnadu_crops.csv")
            self.rainfall = pd.read_csv("final_rainfall_2021_25.csv")
            self.nutrients = pd.read_csv("crop_nutrients.csv")

            # Clean and prepare the data
            self._clean_data()

            # Create dataset
            self.dataset = self._create_training_dataset()
            
            # Verify required columns exist
            required_columns = ['District', 'Seasons', 'Crop', 'Duration']
            for col in required_columns:
                if col not in self.dataset.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Encode categorical fields
            self.dataset['District_encoded'] = self.le_district.fit_transform(self.dataset['District'])
            self.dataset['Season_encoded'] = self.le_season.fit_transform(self.dataset['Seasons'])
            self.dataset['Crop_encoded'] = self.le_crop.fit_transform(self.dataset['Crop'])
            self.dataset['Duration_encoded'] = self.le_duration.fit_transform(self.dataset['Duration'].astype(str))

            # Features and target
            features = ['District_encoded', 'Season_encoded', 'Duration_encoded', 'Irrigation', 'Rainfall', 'N', 'P', 'K']
            target = 'Crop_encoded'

            X = self.dataset[features]
            y = self.dataset[target]

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)

            print("✅ Model trained successfully.")

        except Exception as e:
            print(f"❌ Error during initialization: {str(e)}")
            raise  # Re-raise the exception for debugging

    def _clean_data(self):
        # Clean column names
        self.crops.columns = self.crops.columns.str.strip().str.lower()
        self.rainfall.columns = self.rainfall.columns.str.strip().str.lower()
        self.nutrients.columns = self.nutrients.columns.str.strip().str.lower()

        # Clean district names
        self.rainfall['district'] = self.rainfall['district'].str.strip().str.title()
        self.nutrients['district'] = self.nutrients['district'].str.strip().str.title()

        # Handle duration
        self.crops['duration'] = pd.to_numeric(self.crops['duration'], errors='coerce').fillna(90)

        # Calculate total rainfall
        rainfall_cols = [
            'actual rainfall in south west monsoon (june\'yyyy to september\'yyyy) in mm',
            'actual rainfall in north east monsoon (october\'yyyy to december\'yyyy) in mm',
            'actual rainfall in winter season (january\'yyyy to february\'yyyy) in mm',
            'actual rainfall in hot weather season (march\'yyyy to may\'yyyy) in mm'
        ]
        
        # Only use columns that exist in the dataframe
        available_cols = [col for col in rainfall_cols if col in self.rainfall.columns]
        self.rainfall['rainfall'] = self.rainfall[available_cols].sum(axis=1)

    def _create_training_dataset(self):
        all_combinations = []
        for _, rain_row in self.rainfall.iterrows():
            district = rain_row['district']
            rainfall = rain_row['rainfall']

            nutrient_row = self.nutrients[self.nutrients['district'] == district]
            if nutrient_row.empty:
                continue

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

                # Calculate suitability score (0-100)
                rain_score = self._calculate_score(rainfall, min_rain, max_rain)
                n_score = self._calculate_score(N, min_N, min_N*1.5)
                p_score = self._calculate_score(P, min_P, min_P*1.5)
                k_score = self._calculate_score(K, min_K, min_K*1.5)
                
                # Weighted average
                total_score = (rain_score*0.4 + n_score*0.2 + p_score*0.2 + k_score*0.2)

                for season in ['Kharif', 'Rabi', 'Summer', 'Winter']:
                    season_match = 1 if season.lower() in str(crop_row.get('season', '')).lower() else 0.5
                    total_score *= season_match
                    
                    for irrigation in [0, 1]:
                        irrigation_match = 1 if irrigation == crop_row.get('irrigation', 0) else 0.7
                        final_score = total_score * irrigation_match
                        
                        all_combinations.append({
                            'District': district,
                            'Seasons': season,
                            'Crop': crop,
                            'Duration': duration,
                            'Irrigation': irrigation,
                            'Rainfall': rainfall,
                            'N': N,
                            'P': P,
                            'K': K,
                            'Score': final_score
                        })

        return pd.DataFrame(all_combinations)

    def _calculate_score(self, value, min_val, max_val):
        """Calculate a score between 0-100 based on how close the value is to optimal range"""
        if value < min_val:
            return 50 * (value / min_val) if min_val != 0 else 0
        elif value > max_val:
            return max(0, 100 - (value - max_val) / max_val * 50) if max_val != 0 else 0
        else:
            return 100

    def _get_season(self, month):
        month = str(month).lower()
        if month in ['june', 'july', 'august', 'september']:
            return 'Kharif'
        elif month in ['october', 'november', 'december']:
            return 'Rabi'
        elif month in ['march', 'april', 'may']:
            return 'Summer'
        elif month in ['january', 'february']:
            return 'Winter'
        return 'Kharif'  # Default season

    def recommend(self, district, month, irrigation, duration_label):
        try:
            duration_mapping = {
                'short': 90,
                'medium': 150,
                'long': 240
            }

            duration_days = duration_mapping.get(duration_label.lower(), 150)  # Default to medium
            season = self._get_season(month)
            district = district.strip().title()

            # Get required data
            rainfall = self._get_rainfall(district)
            N, P, K = self._get_nutrients(district)

            # Encode inputs
            district_encoded = self.le_district.transform([district])[0]
            season_encoded = self.le_season.transform([season])[0]
            duration_encoded = self.le_duration.transform([str(duration_days)])[0]

            input_features = [
                district_encoded,
                season_encoded,
                duration_encoded,
                int(irrigation),
                rainfall,
                N, P, K
            ]

            # Get probabilities for all crops
            probas = self.model.predict_proba([input_features])[0]
            
            recommendations = []
            for crop_idx, prob in enumerate(probas):
                crop = self.le_crop.inverse_transform([crop_idx])[0]
                score = self._calculate_crop_suitability(
                    crop, district, season, irrigation, 
                    rainfall, N, P, K, duration_days
                )
                
                final_score = (score * 0.7) + (prob * 100 * 0.3)
                crop_info = self.crops[self.crops['crop'] == crop].iloc[0]
                
                recommendations.append({
                    "crop": crop,
                    "score": round(final_score, 1),
                    "duration": f"{duration_days} days",
                    "district": district,
                    "season": season,
                    "description": self._generate_description(crop, district, season, final_score, 
                                                         rainfall, N, P, K, duration_days)
                })

            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                "recommendations": recommendations[:5],
                "conditions": {
                    "district": district,
                    "season": season,
                    "rainfall": f"{rainfall:.1f} mm",
                    "soil_nutrients": f"N: {N:.1f}%, P: {P:.1f}%, K: {K:.1f}%",
                    "irrigation": "Available" if irrigation else "Not Available",
                    "duration": f"{duration_days} days"
                }
            }

        except Exception as e:
            return {"error": f"Recommendation failed: {str(e)}"}

    def _calculate_crop_suitability(self, crop, district, season, irrigation, rainfall, N, P, K, duration_days):
        """Calculate suitability score for a specific crop"""
        crop_row = self.crops[self.crops['crop'] == crop].iloc[0]
        
        min_rain = crop_row['min_rain']
        max_rain = crop_row['max_rain']
        min_N = crop_row['min_n']
        min_P = crop_row['min_p']
        min_K = crop_row['min_k']
        ideal_duration = crop_row['duration']
        
        rain_score = self._calculate_score(rainfall, min_rain, max_rain)
        n_score = self._calculate_score(N, min_N, min_N*1.5)
        p_score = self._calculate_score(P, min_P, min_P*1.5)
        k_score = self._calculate_score(K, min_K, min_K*1.5)
        
        duration_diff = abs(duration_days - ideal_duration)
        duration_score = max(0, 100 - (duration_diff / ideal_duration * 50))
        
        season_match = 1 if season.lower() in str(crop_row.get('season', '')).lower() else 0.5
        irrigation_match = 1 if irrigation == crop_row.get('irrigation', 0) else 0.7
        
        total_score = (
            rain_score * 0.3 +
            n_score * 0.15 +
            p_score * 0.15 +
            k_score * 0.15 +
            duration_score * 0.15 +
            (season_match * 0.05) +
            (irrigation_match * 0.05)
        )
        
        return min(100, max(0, total_score))

    def _generate_description(self, crop, district, season, score, rainfall, N, P, K, duration):
        crop_info = self.crops[self.crops['crop'] == crop].iloc[0]
        
        reasons = []
        rainfall_status = "optimal" if (crop_info['min_rain'] <= rainfall <= crop_info['max_rain']) else \
                        "low" if rainfall < crop_info['min_rain'] else "high"
        reasons.append(f"rainfall is {rainfall_status} ({rainfall:.1f}mm vs ideal {crop_info['min_rain']}-{crop_info['max_rain']}mm)")
        
        nutrients = []
        if N < crop_info['min_n']:
            nutrients.append(f"low N ({N:.1f}% < {crop_info['min_n']}%)")
        if P < crop_info['min_p']:
            nutrients.append(f"low P ({P:.1f}% < {crop_info['min_p']}%)")
        if K < crop_info['min_k']:
            nutrients.append(f"low K ({K:.1f}% < {crop_info['min_k']}%)")
        if not nutrients:
            nutrients.append("adequate nutrients")
        reasons.append(", ".join(nutrients))
        
        if season.lower() in str(crop_info.get('season', '')).lower():
            reasons.append("good season match")
        else:
            reasons.append("suboptimal season")
            
        duration_diff = abs(duration - crop_info['duration'])
        if duration_diff <= 30:
            reasons.append("good duration match")
        else:
            reasons.append(f"duration differs by {duration_diff} days from ideal")
            
        irrigation_match = "matches" if int(crop_info.get('irrigation', 0)) else "is optional"
        reasons.append(f"irrigation {irrigation_match} crop preference")
        
        return (f"{crop} scored {score:.1f}/100 because: " + 
                "; ".join(reasons))

    def _get_rainfall(self, district):
        row = self.rainfall[self.rainfall['district'] == district]
        return row['rainfall'].values[0] if not row.empty else 0

    def _get_nutrients(self, district):
        row = self.nutrients[self.nutrients['district'] == district]
        if not row.empty:
            return row['n_high_%'].values[0], row['p_high_%'].values[0], row['k_high_%'].values[0]
        return 0, 0, 0
