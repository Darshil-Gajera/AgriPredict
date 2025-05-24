import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class Category3Predictor:
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.train_model()
        
    def train_model(self):
        try:
            # Load historical data
            df = pd.read_csv('static/data/categories/category-3_collegewise_merit.csv')
            
            # Prepare features and labels
            X = []
            y = []
            
            # Generate synthetic training data
            for _, row in df.iterrows():
                for category in ['GENERAL', 'SEBC', 'SC', 'ST', 'EWS']:
                        cutoff = float(row[category])
                        for offset in [-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10]:
                            merit = cutoff + offset
                            X.append([
                                merit,
                                cutoff,
                                self.category_to_numeric(category),
                                merit - cutoff,
                                1 if merit >= cutoff else 0
                            ])
                            # Probability distribution
                            if merit >= cutoff + 7:
                                y.append(0.95)
                            elif merit >= cutoff + 3:
                                y.append(0.8)
                            elif merit >= cutoff:
                                y.append(0.6)
                            elif merit >= cutoff - 5:
                                y.append(0.4)
                            elif merit >= cutoff - 10:
                                y.append(0.2)
                            else:
                                y.append(0.05)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            # Train model
            self.rf_model.fit(X, y)
            
        except Exception as e:
            print(f"Error in training model: {str(e)}")
            
    def category_to_numeric(self, category):
        category_map = {
            'GENERAL': 0,
            'SEBC': 1,
            'SC': 2,
            'ST': 3,
            'EWS': 4
        }
        return category_map.get(category, 0)
        
    def prepare_features(self, theory_marks, total_theory_marks, gujcet_marks, subject_group, category, has_farming):
        # Normalize theory marks to percentage (60% weightage)
        theory_percentage = (theory_marks / total_theory_marks) * 100
        theory_contribution = theory_percentage * 0.60
        
        # Calculate GUJCET percentage (40% weightage)
        gujcet_percentage = (gujcet_marks / 120) * 100
        gujcet_contribution = gujcet_percentage * 0.40
        
        # Calculate base merit
        merit = theory_contribution + gujcet_contribution
                
        # Add farming background bonus after all other calculations
        if has_farming:
            merit += 5  # 5% bonus for farming background
                
        return round(merit, 2), self.scaler.transform([[merit, 0, self.category_to_numeric(category), 0, 1]])

    def predict_admission_chance(self, merit, cutoff, category):
        try:
            diff = merit - cutoff
            
            if diff >= 7:
                prob = 0.95
            elif diff >= 4:
                prob = 0.90 + ((diff - 4) * 0.02)
            elif diff >= 2:
                prob = 0.80 + ((diff - 2) * 0.05)
            elif diff >= 0:
                prob = 0.70 + (diff * 0.05)
            elif diff >= -3:
                prob = 0.50 + ((diff + 3) * 0.067)
            elif diff >= -5:
                prob = 0.30 + ((diff + 5) * 0.04)
            else:
                prob = max(0.10, 0.30 + (diff * 0.04))
                
            prob_percentage = round(prob * 100, 2)
            
            if prob >= 0.85:
                return {
                    'chance': 'High',
                    'probability': prob_percentage,
                    'round_prediction': '1st Round - Seat Allocation Likely'
                }
            elif prob >= 0.65:
                return {
                    'chance': 'Medium',
                    'probability': prob_percentage,
                    'round_prediction': '1st/2nd Round - Good Possibility'
                }
            elif prob >= 0.45:
                return {
                    'chance': 'Low',
                    'probability': prob_percentage,
                    'round_prediction': '2nd/3rd Round - Wait Required'
                }
            else:
                return {
                    'chance': 'Very Low',
                    'probability': prob_percentage,
                    'round_prediction': 'Consider Other Options'
                }
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'chance': 'Unknown',
                'probability': 0,
                'round_prediction': 'Unable to predict'
            }
            
    def predict_colleges(self, merit, features_scaled, category, subject_group):
        try:
            df = pd.read_csv('static/data/categories/category-3_collegewise_merit.csv')
            eligible_colleges = []
            
            for _, row in df.iterrows():
                    
                cutoff = float(row[category])
                prediction = self.predict_admission_chance(merit, cutoff, category)
                
                college_name = row['COLLEGE NAME']
                course = str(row['COURSE']).upper()
                location = str(college_name).split(',')[-1].strip() if ',' in str(college_name) else 'Unknown'
                
                # Updated course filtering logic
                is_eligible = False
                if subject_group == 'PCB':
                    is_eligible = any(keyword in course for keyword in ['NUTRITION', 'FOOD', 'BIOLOGY', 'B.SC'])
                elif subject_group == 'PCM':
                    # Updated PCM eligibility to include relevant courses
                    is_eligible = any(keyword in course for keyword in ['MATHEMATICS', 'PHYSICS', 'CHEMISTRY', 'B.SC'])
                    
                if is_eligible:
                    eligible_colleges.append({
                        'name': college_name,
                        'course': row['COURSE'],
                        'location': location,
                        'cutoff': cutoff,
                        'chance': prediction['chance'],
                        'probability': prediction['probability'],
                        'round_prediction': prediction['round_prediction']
                    })
            
            # Sort by probability descending
            eligible_colleges.sort(key=lambda x: -x['probability'])
            
            return eligible_colleges
            
        except Exception as e:
            print(f"Error predicting colleges: {str(e)}")
            return []

    def predict(self, theory_marks, total_theory_marks, gujcet_marks, category, subject_group, has_farming=False):
        try:
            # Calculate merit score
            merit, features_scaled = self.prepare_features(
                theory_marks=theory_marks,
                total_theory_marks=total_theory_marks,
                gujcet_marks=gujcet_marks,
                subject_group=subject_group,
                category=category,
                has_farming=has_farming
            )
            
            # Get college recommendations
            recommendations = self.predict_colleges(merit, features_scaled, category, subject_group)
            
            return {
                'merit_score': merit,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'merit_score': 0,
                'recommendations': []
            }