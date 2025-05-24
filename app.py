from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from category2_predictor import Category2Predictor  # Import the Category2Predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agripage.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Add this line to suppress warning
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    notifications = Notification.query.filter_by(is_active=True).order_by(Notification.created_at.desc()).limit(5).all()
    return render_template('home.html', notifications=notifications)

@app.route('/predict/<category>')
def predict(category):
    if category in ["category1", "category-1"]:
        return render_template('merit_calculator.html')
    
    # Load your trained model (you'll need to create and train these first)
    # Example using pickle to load a saved model:
    # model = pickle.load(open(f'models/{category}_model.pkl', 'rb'))
    
    # Get input features from request
    # features = request.args.get('features')
    
    # Preprocess features
    # processed_features = preprocess_features(features)
    
    # Make prediction
    # prediction = model.predict(processed_features)
    
    return render_template(f'predict_{category}.html')

class CollegePredictor:
    def __init__(self):
        # Balanced hyperparameters to prevent overfitting
        self.rf_model = RandomForestClassifier(
            n_estimators=300,          # Reduced from 500 to prevent overfitting
            max_depth=8,               # Reduced from 15 to prevent overfitting
            min_samples_split=10,      # Increased to ensure better generalization
            min_samples_leaf=5,        # Increased to prevent overfitting
            max_features='sqrt',       # Add feature selection
            class_weight='balanced',   # Keep balanced weights
            random_state=42
        )
        self.scaler = StandardScaler()
        self.train_model()
        
    def train_model(self):
        try:
            # Load historical data
            df = pd.read_csv('static/data/categories/category-1_collegewise_merit.csv')
            
            # Prepare features and labels with more balanced data
            X = []
            y = []
            
            # Generate synthetic training data with better distribution
            for _, row in df.iterrows():
                for category in ['GENERAL', 'SEBC', 'SC', 'ST', 'EWS', 'PH-VH', 'Ex - Serv.']:
                    if pd.notna(row[category]):
                        cutoff = float(row[category])
                        # More balanced offset distribution
                        for offset in [-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10]:
                            merit = cutoff + offset
                            # Simplified feature set to prevent overfitting
                            X.append([
                                merit,                    # Merit score
                                cutoff,                   # College cutoff
                                self.category_to_numeric(category),  # Category
                                merit - cutoff,           # Merit-cutoff difference
                                1 if merit >= cutoff else 0  # Above cutoff flag
                            ])
                            # Smoother probability distribution
                            if merit >= cutoff + 7:
                                y.append(0.95)    # Very high chance
                            elif merit >= cutoff + 3:
                                y.append(0.8)     # High chance
                            elif merit >= cutoff:
                                y.append(0.6)     # Medium chance
                            elif merit >= cutoff - 5:
                                y.append(0.4)     # Low chance
                            elif merit >= cutoff - 10:
                                y.append(0.2)     # Very low chance
                            else:
                                y.append(0.05)    # Minimal chance
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            # Split with stratification and larger test size
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, 
                stratify=np.digitize(y, bins=[0.2, 0.4, 0.6, 0.8])
            )
            
            # Train model
            self.rf_model.fit(X_train, y_train)
            
            # Calculate and print accuracy metrics
            train_score = self.rf_model.score(X_train, y_train)
            test_score = self.rf_model.score(X_test, y_test)
            print(f"Training accuracy: {train_score:.4f}")
            print(f"Testing accuracy: {test_score:.4f}")
            
            # Check for overfitting
            if train_score - test_score > 0.1:
                print("Warning: Model might be overfitting")
            
        except Exception as e:
            print(f"Error in training model: {str(e)}")
            
    def category_to_numeric(self, category):
        category_map = {
            'GENERAL': 0, 'SEBC': 1, 'SC': 2, 'ST': 3,
            'EWS': 4, 'Other Board': 5, 'PH-VH': 6, 'Ex - Serv.': 7
        }
        return category_map.get(category, 0)
        
    def prepare_features(self, theory_marks, total_theory_marks, gujcet_marks, category, has_farming):
        # Normalize theory marks to 300 (if not already)
        if total_theory_marks != 300:
            normalized_theory = (theory_marks / total_theory_marks) * 300
        else:
            normalized_theory = theory_marks
        
        # Calculate merit using 60-40 formula
        theory_percentage = (normalized_theory / 300) * 100
        gujcet_percentage = (gujcet_marks / 120) * 100
        
        # Calculate final merit (60% theory + 40% GUJCET)
        merit = (theory_percentage * 0.60) + (gujcet_percentage * 0.40)
        
        # Add farming background bonus
        if has_farming:
            merit += 5
            
        return merit

    def predict_admission_chance(self, merit, cutoff, category):
        try:
            # Calculate merit difference percentage
            diff_percentage = merit - cutoff
            
            # More accurate probability calculation based on merit-cutoff difference
            if diff_percentage >= 2:
                prob = 0.90  # Very high chance (90%)
                chance_text = "High"
                round_text = "1st Round - High Chance (90%)"
            elif diff_percentage >= 0:
                prob = 0.70  # Good chance (70%)
                chance_text = "Good"
                round_text = "1st Round - Good Chance (70%)"
            elif diff_percentage >= -3:
                prob = 0.40  # Medium chance (40%)
                chance_text = "Medium"
                round_text = "1st Round - Medium Chance (40%)"
            elif diff_percentage >= -7:
                prob = 0.25  # Low chance (25%)
                chance_text = "Low"
                round_text = "1st Round - Low Chance (25%)"
            elif diff_percentage >= -10:
                prob = 0.15  # Very low chance (15%)
                chance_text = "Very Low"
                round_text = "1st Round - Low Chance (15%)"
            else:
                # More granular minimal chances based on difference
                base_prob = 0.10
                # Further reduce probability for larger differences
                reduction = min(0.05, abs(diff_percentage - 10) * 0.005)
                prob = max(0.05, base_prob - reduction)  # Don't go below 5%
                chance_text = "Minimal"
                round_text = f"1st Round - Minimal Chance ({round(prob * 100)}%)"
            
            # Apply category-based adjustments
            category_adjustments = {
                'GENERAL': 1.0,
                'SEBC': 1.05,
                'SC': 1.1,
                'ST': 1.15,
                'EWS': 1.05,
                'PH-VH': 1.2,
                'Ex - Serv.': 1.1
            }
            
            # Apply category adjustment
            adj_factor = category_adjustments.get(category, 1.0)
            prob = min(0.95, prob * adj_factor)  # Cap at 95% probability
            
            prob_percentage = round(prob * 100, 2)
            
            return {
                'chance': chance_text,
                'probability': prob_percentage,
                'round_prediction': round_text
            }
                
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'chance': 'Unknown',
                'probability': 0,
                'round_prediction': 'Unable to predict'
            }

def category_to_numeric(category):
    category_map = {
        'GENERAL': 0, 'SEBC': 1, 'SC': 2, 'ST': 3,
        'EWS': 4, 'Other Board': 5, 'PH-VH': 6, 'Ex - Serv.': 7
    }
    return category_map.get(category, 0)

@app.route('/calculate_merit', methods=['POST'])
def calculate_merit():
    if not request.is_json:
        return jsonify({'error': 'Invalid request format'}), 400
        
    try:
        data = request.get_json()
        
        # Extract and validate form data
        theory_marks = float(data.get('theory_marks', 0))
        total_theory_marks = float(data.get('total_theory_marks', 0))
        gujcet_marks = float(data.get('gujcet_marks', 0))
        category = data.get('category', '')
        has_farming = data.get('farming_background', False)
        
        # Validate inputs
        if not all([theory_marks, total_theory_marks, gujcet_marks, category]):
            return jsonify({'error': 'All fields are required'}), 400
            
        if theory_marks > total_theory_marks:
            return jsonify({'error': 'Theory marks cannot exceed total marks'}), 400
            
        if gujcet_marks > 120:
            return jsonify({'error': 'GUJCET marks cannot exceed 120'}), 400
            
        predictor = CollegePredictor()
        merit = predictor.prepare_features(
            theory_marks=theory_marks,
            total_theory_marks=total_theory_marks,
            gujcet_marks=gujcet_marks,
            category=category,
            has_farming=has_farming
        )
        
        # Get college predictions with proper error handling
        try:
            df = pd.read_csv('static/data/categories/category-1_collegewise_merit.csv')
            eligible_colleges = []
            
            for _, row in df.iterrows():
                if pd.isna(row[category]):
                    continue
                    
                cutoff = float(row[category])
                prediction = predictor.predict_admission_chance(merit, cutoff, category)
                
                college_name = row['COLLEGE NAME']
                location = str(college_name).split(',')[-1].strip() if ',' in str(college_name) else 'Unknown'
                
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
            
            return jsonify({
                'merit': round(merit, 2),
                'colleges': eligible_colleges
            })
            
        except FileNotFoundError:
            return jsonify({'error': 'College data not found'}), 500
        except Exception as e:
            print(f"Error processing college data: {str(e)}")
            return jsonify({'error': 'Error processing college predictions'}), 500
            
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error calculating merit: {str(e)}")
        return jsonify({'error': 'An error occurred during calculation'}), 500

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/college/<int:college_id>')
def college_details(college_id):
    return render_template('college_details.html')

@app.route('/admission-process')
def admission_process():
    return render_template('admission_process.html')

@app.route('/notifications')
def notifications():
    notifications = [
        {
            'date': '17/05/2025',
            'title': 'સ્નાતક કક્ષાના અભ્યાસક્રમોમાં પ્રવેશ અંગેની જાહેરાત-૨૦૨૫-૨૬',
            'description': 'Important notification regarding admissions for undergraduate programs',
            'pdf_file': 'admission_announcement_2025_26.pdf'  # Matches exact filename
        },
        {
            'date': '15/05/2025',
            'title': 'સ્નાતક કક્ષાના અભ્યાસક્રમોમાં પ્રવેશ અંગેની માહિતી પુસ્તિકા-૨૦૨૫-૨૬',
            'description': 'Information booklet for undergraduate program admissions',
            'pdf_file': 'admission_information_booklet_2025_26.pdf'  # Matches exact filename
        },
        {
            'date': '12/05/2025',
            'title': 'ધોરણ-૧૨ (વિજ્ઞાન પ્રવાહ) પછીના સ્નાતક કક્ષાના અભ્યાસક્રમોમાં પ્રવેશ માટે ફોર્સ ભરતી વખતે અપલોડ રચઍડના થતા અન્યત્ર પ્રમાણપત્રોની વિગત.',
            'description': 'Details of certificates to be uploaded during form submission',
            'pdf_file': 'required_certificates_details_2025_26.pdf'  # Matches exact filename
        }
    ]
    return render_template('notifications.html', notifications=notifications)

@app.route('/api/comparison-data')
def get_comparison_data():
    try:
        # Read data from CSV file
        df = pd.read_csv('static/data/categories/category-1_collegewise_merit.csv')
        
        # Extract unique cities, courses, and colleges
        cities = sorted(list(set(college.split(',')[-1].strip() 
                    for college in df['COLLEGE NAME'] if ',' in college)))
        courses = sorted(list(set(df['COURSE'])))
        colleges = sorted(list(set(df['COLLEGE NAME'])))
        
        return jsonify({
            'cities': [{'id': i, 'name': city} for i, city in enumerate(cities)],
            'courses': [{'id': i, 'name': course} for i, course in enumerate(courses)],
            'colleges': [{'id': i, 'name': college} for i, college in enumerate(colleges)]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_colleges():
    try:
        data = request.get_json()
        df = pd.read_csv('static/data/categories/category-1_collegewise_merit.csv')
        
        # Filter based on selections
        filtered_data = []
        for _, row in df.iterrows():
            college_city = row['COLLEGE NAME'].split(',')[-1].strip() if ',' in row['COLLEGE NAME'] else ''
            
            if ((not data['cities'] or college_city in data['cities']) and
                (not data['courses'] or row['COURSE'] in data['courses']) and
                (not data['colleges'] or row['COLLEGE NAME'] in data['colleges'])):
                
                filtered_data.append({
                    'name': row['COLLEGE NAME'],
                    'city': college_city,
                    'course': row['COURSE'],
                    'cutoff': row['GENERAL'],  # You might want to handle different categories
                    'seats': 'Available',  # You might want to add actual seat data
                    'facilities': 'Standard Facilities'  # You might want to add actual facilities data
                })
        
        return jsonify(filtered_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scholarships')
def scholarships():
    scholarships = [
        {
            'Scholarship Name': 'MYSY Scholarship (Mukhyamantri Yuva Swavalamban Yojana)',
            'Eligible Courses': 'B.Sc. (Hons.) Agriculture, Horticulture, Forestry, Community Science, Food Nutrition & Dietetics; B.Tech. in Agricultural Engg., Food Tech, Biotechnology, Agri IT, Renewable Energy',
            'Eligibility Criteria': 'Gujarat domicile, 80%+ in 12th (Science stream), family income ≤ ₹6 lakh/year',
            'Benefits': '₹50,000 tuition + hostel & book assistance',
            'Application Platform': 'https://mysy.guj.nic.in',
            'Remarks': 'Major state scholarship for professional UG students'
        },
        {
            'Scholarship Name': 'National Scholarship Portal (NSP)',
            'Eligible Courses': 'All UG courses including B.Sc. (Hons.) and B.Tech. (Agri & allied)',
            'Eligibility Criteria': 'Varies by scheme: SC/ST/OBC/EWS/Minority; income limit applies',
            'Benefits': '₹10,000–₹30,000+ (tuition + maintenance)',
            'Application Platform': 'https://scholarships.gov.in',
            'Remarks': 'Includes Post-Matric, Top Class schemes'
        },
        {
            'Scholarship Name': 'ICAR National Talent Scholarship (NTS)',
            'Eligible Courses': 'B.Sc. (Hons.) in Agriculture / Horticulture / Forestry (ICAR-AU admitted students only)',
            'Eligibility Criteria': 'Admission through ICAR AIEEA UG, non-domicile of that state',
            'Benefits': '₹2,000/month (₹24,000/year)',
            'Application Platform': 'Via ICAR-AU during admission',
            'Remarks': 'Only for ICAR seat holders'
        },
        {
            'Scholarship Name': 'Digital Gujarat Scholarship',
            'Eligible Courses': 'All UG courses (B.Sc. / B.Tech. – Agriculture & allied fields) for SC/ST/OBC/SEBC/EWS',
            'Eligibility Criteria': 'Gujarat domicile; income ₹2L–₹6L (as per category)',
            'Benefits': '₹5,000–₹20,000+ based on course/category',
            'Application Platform': 'https://digitalgujarat.gov.in',
            'Remarks': 'State-run social justice scholarships'
        },
        {
            'Scholarship Name': 'Jain/ Muslim Minority Scholarships (NSP or NGOs)',
            'Eligible Courses': 'All UG courses including Agriculture',
            'Eligibility Criteria': 'Minority students with 50%+ marks, income ≤ ₹2 lakh',
            'Benefits': '₹5,000–₹25,000+',
            'Application Platform': 'https://scholarships.gov.in / NGO portals',
            'Remarks': 'Check with local Minority Welfare Dept.'
        },
        {
            'Scholarship Name': 'Kanya Kelavani Yojana (Girls only)',
            'Eligible Courses': 'All UG courses including Agri & Allied Sciences',
            'Eligibility Criteria': 'Gujarat domicile, girl child',
            'Benefits': '₹5,000–₹25,000+',
            'Application Platform': 'https://digitalgujarat.gov.in',
            'Remarks': 'For promotion of girl education'
        },
        {
            'Scholarship Name': 'INSPIRE Scholarship (by DST)',
            'Eligible Courses': 'B.Sc. (Hons.) Agriculture, Biotechnology, Pure Science',
            'Eligibility Criteria': 'Top 1% in 12th (Science stream), pursuing UG in science',
            'Benefits': '₹80,000/year (up to 5 years)',
            'Application Platform': 'https://online-inspire.gov.in',
            'Remarks': 'Central Govt. for top science rankers'
        },
        {
            'Scholarship Name': 'ONGC Scholarship for SC/ST/EWS',
            'Eligible Courses': 'B.Tech. Agricultural Engg., Food Tech, Biotechnology; B.Sc. Agriculture & allied',
            'Eligibility Criteria': 'SC/ST/EWS, 1st-year UG, 60%+ in 12th, income ≤ ₹2 lakh',
            'Benefits': '₹48,000/year',
            'Application Platform': 'https://ongcindia.com',
            'Remarks': 'National-level limited scholarship'
        },
        {
            'Scholarship Name': 'Sitaram Jindal Foundation Scholarship',
            'Eligible Courses': 'All UG courses including B.Sc. / B.Tech. (Agri & Allied)',
            'Eligibility Criteria': 'Income ≤ ₹2.5 lakh (general), good academic record',
            'Benefits': '₹2,000–₹2,500/month',
            'Application Platform': 'https://sjfoundation.org',
            'Remarks': 'Private NGO scholarship'
        },
        {
            'Scholarship Name': 'LIC Golden Jubilee Scholarship',
            'Eligible Courses': 'All UG courses',
            'Eligibility Criteria': '12th pass, 60%+ marks, income ≤ ₹2 lakh/year',
            'Benefits': '₹10,000–₹20,000/year',
            'Application Platform': 'https://licindia.in',
            'Remarks': 'For economically weaker students'
        }
    ]
    return render_template('scholarships.html', scholarships=scholarships)

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('register.html')

@app.route('/calculator')
def calculator():
    return render_template('calculator.html')

@app.context_processor
def inject_now():
    from datetime import datetime, UTC
    return {'now': datetime.now(UTC)}

@app.route('/predict_category2', methods=['POST'])
def predict_category2():
    try:
        data = request.get_json()
        
        # Initialize predictor
        predictor = Category2Predictor()
        
        # Get prediction
        result = predictor.predict(
            theory_marks=data['theory_marks'],
            total_theory_marks=data['total_theory_marks'],
            gujcet_marks=data['gujcet_marks'],
            category=data['category'],
            subject_group=data['subject_group']
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_category3', methods=['POST'])
def predict_category3():
    if not request.is_json:
        return jsonify({'error': 'Invalid request format'}), 400
        
    try:
        data = request.get_json()
        
        # Extract and validate form data
        theory_marks = float(data.get('theory_marks', 0))
        total_theory_marks = float(data.get('total_theory_marks', 0))
        gujcet_marks = float(data.get('gujcet_marks', 0))
        category = data.get('category', '')
        subject_group = data.get('subject_group', '')
        has_farming = data.get('farming_background', False)
        
        # Validate inputs
        if not all([theory_marks, total_theory_marks, gujcet_marks, category, subject_group]):
            return jsonify({'error': 'All fields are required'}), 400
            
        if theory_marks > total_theory_marks:
            return jsonify({'error': 'Theory marks cannot exceed total marks'}), 400
            
        if gujcet_marks > 120:
            return jsonify({'error': 'GUJCET marks cannot exceed 120'}), 400
            
        # Initialize predictor and get predictions
        from createcategory3_predictor import Category3Predictor
        predictor = Category3Predictor()
        result = predictor.predict(
            theory_marks=theory_marks,
            total_theory_marks=total_theory_marks,
            gujcet_marks=gujcet_marks,
            category=category,
            subject_group=subject_group,
            has_farming=has_farming
        )
        
        return jsonify({
            'merit_score': result['merit_score'],
            'recommendations': result['recommendations']
        })
            
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"Error calculating merit: {str(e)}")
        return jsonify({'error': 'An error occurred during calculation'}), 500

@app.route('/college-list')
def college_list():
    return render_template('college_list.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)