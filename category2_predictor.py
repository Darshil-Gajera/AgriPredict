import pandas as pd
import numpy as np

class Category2Predictor:
    def __init__(self):
        # Load the college data once during initialization
        try:
            self.df = pd.read_csv('static/data/categories/category-2_collegewise_merit.csv')
        except Exception as e:
            print(f"Error loading college data: {str(e)}")
            self.df = None

    def predict(self, theory_marks, total_theory_marks, gujcet_marks, category, subject_group, 
                selected_cities=None, selected_courses=None, selected_colleges=None, sort_by_merit=False):
        try:
            # Input validation
            if not all(isinstance(x, (int, float)) for x in [theory_marks, total_theory_marks, gujcet_marks]):
                raise ValueError("Invalid input: Marks must be numeric values")
            
            if theory_marks > total_theory_marks:
                raise ValueError("Theory marks cannot be greater than total marks")
                
            if gujcet_marks > 120:
                raise ValueError("GUJCET marks cannot exceed 120")
            
            # Calculate percentage of theory marks (60% weightage)
            theory_percentage = (theory_marks / total_theory_marks) * 100
            theory_contribution = theory_percentage * 0.60
            
            # Calculate GUJCET percentage (40% weightage)
            gujcet_percentage = (gujcet_marks / 120) * 100
            gujcet_contribution = gujcet_percentage * 0.40
            
            # Calculate final merit score
            merit_score = round(theory_contribution + gujcet_contribution, 2)
            
            # Get college recommendations with filters
            if self.df is not None:
                colleges = self._get_college_recommendations(
                    merit_score, category, subject_group,
                    selected_cities, selected_courses, selected_colleges, sort_by_merit
                )
            else:
                colleges = []
            
            return {
                'success': True,
                'merit_score': merit_score,
                'recommendations': colleges
            }
            
        except Exception as e:
            print(f"Error calculating merit: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'merit_score': 0,
                'recommendations': []
            }
    
    def _get_college_recommendations(self, merit_score, category, subject_group,
                                   selected_cities=None, selected_courses=None, 
                                   selected_colleges=None, sort_by_merit=False):
        recommendations = []
        
        # Create a copy of the dataframe to apply filters
        filtered_df = self.df.copy()
        
        # Apply filters if provided
        if selected_cities:
            filtered_df = filtered_df[filtered_df['LOCATION'].isin(selected_cities)]
        
        if selected_courses:
            filtered_df = filtered_df[filtered_df['COURSE'].isin(selected_courses)]
            
        if selected_colleges:
            filtered_df = filtered_df[filtered_df['COLLEGE NAME'].isin(selected_colleges)]
        
        for idx, row in filtered_df.iterrows():
            if pd.isna(row[category]):
                continue
                
            cutoff = float(row[category])
            college_name = str(row['COLLEGE NAME']).strip()
            course = str(row['COURSE']).strip()
            location = str(row['LOCATION']).strip() if 'LOCATION' in row else college_name.split(',')[-1].strip()
            
            # Calculate admission chance
            diff = merit_score - cutoff
            
            if subject_group == 'PCB':
                if diff >= 3:
                    chance, probability = "High", 90
                elif diff >= -2:
                    chance, probability = "Medium", 70
                elif diff >= -7:
                    chance, probability = "Low", 40
                else:
                    chance, probability = "Very Low", 20
            else:  # PCM
                if diff >= 5:
                    chance, probability = "High", 90
                elif diff >= 0:
                    chance, probability = "Medium", 70
                elif diff >= -5:
                    chance, probability = "Low", 40
                else:
                    chance, probability = "Very Low", 20
            
            # Get round prediction
            if probability >= 80:
                round_prediction = "1st Round - High Chance"
            elif probability >= 60:
                round_prediction = "1st/2nd Round - Good Possibility"
            else:
                round_prediction = "Later Rounds - Wait Recommended"
            
            recommendations.append({
                'college_name': college_name,
                'course': course,
                'location': location,
                'cutoff': cutoff,
                'chance': chance,
                'probability': probability,
                'round_prediction': round_prediction,
                'merit_score': merit_score
            })
        
        # Sort recommendations
        if sort_by_merit:
            recommendations.sort(key=lambda x: x['cutoff'], reverse=True)
        else:
            recommendations.sort(key=lambda x: x['probability'], reverse=True)
        
        # Add numbering after sorting
        for i, rec in enumerate(recommendations, 1):
            rec['college_name'] = f"{i}. {rec['college_name']}"
        
        return recommendations