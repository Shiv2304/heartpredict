import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Make datetime available globally in Jinja2
app.jinja_env.globals.update(datetime=datetime)

# Load the pre-trained model and scaler
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
model = pickle.load(open('models/rf_model.pkl', 'rb'))

# User model with gender field
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)  # Gender field
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Prediction history model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_data = db.Column(db.String(200), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    date = db.Column(db.DateTime, default=db.func.current_timestamp())

with app.app_context():
    db.create_all()  # Remove or comment out after initial setup if using migrations

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        age = request.form['age']
        gender = request.form['gender']  # Capture gender
        username = request.form['username']
        password = request.form['password']
        terms = request.form.get('terms')
        if not terms:
            flash('You must agree to the terms and conditions')
        elif User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists')
        else:
            new_user = User(name=name, email=email, age=int(age), gender=gender, username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    predictions = Prediction.query.filter_by(user_id=session['user_id']).all()
    positive_count = sum(1 for p in predictions if p.result == 'Positive')
    negative_count = len(predictions) - positive_count
    return render_template('home.html', user=user, predictions=predictions, positive_count=positive_count, negative_count=negative_count)

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.date.desc()).all()
    positive_count = sum(1 for p in predictions if p.result == 'Positive')
    negative_count = len(predictions) - positive_count
    return render_template('dashboard.html', user=user, predictions=predictions, positive_count=positive_count, negative_count=negative_count)

from datetime import datetime

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.date.asc()).all()
    positive_count = sum(1 for p in predictions if p.result == 'Positive')
    negative_count = len(predictions) - positive_count
    # Prepare data for charts
    dates = [pred.date.strftime('%Y-%m-%d') for pred in predictions]
    results = [1 if pred.result == 'Positive' else 0 for pred in predictions]
    return render_template('profile.html', user=user, predictions=predictions, positive_count=positive_count, negative_count=negative_count, dates=dates, results=results)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        input_data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['chest_pain_type']),
            float(request.form['resting_bp']),
            float(request.form['cholesterol']),
            float(request.form['fasting_bs']),
            float(request.form['resting_ecg']),
            float(request.form['max_heart_rate']),
            float(request.form['exercise_angina']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        input_df = pd.DataFrame([input_data], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        pred_record = Prediction(user_id=session['user_id'], input_data=str(input_data), result='Positive' if prediction == 1 else 'Negative')
        db.session.add(pred_record)
        db.session.commit()

        # Ensure minimum 5 suggestions
        suggestions = [
            "Maintain a balanced diet rich in fruits and vegetables.",
            "Engage in regular physical activity (e.g., 30 minutes most days).",
            "Monitor your blood pressure and cholesterol levels regularly.",
            "Avoid smoking and limit alcohol consumption.",
            "Manage stress through relaxation techniques like meditation."
        ]
        if prediction == 1:
            suggestions[0] = "Consult a cardiologist immediately for further evaluation."
            if float(request.form['cholesterol']) > 200:
                suggestions[1] = "Reduce cholesterol with a low-fat diet and consult a doctor."
            if float(request.form['resting_bp']) > 140:
                suggestions[2] = "Monitor blood pressure daily; consider medication if prescribed."
            if float(request.form['fasting_bs']) > 120:
                suggestions[3] = "Manage blood sugar with a balanced diet and regular checkups."

        session['suggestions'] = suggestions  # Store suggestions in session
        return render_template('result.html', prediction=prediction, probability=probability, suggestions=suggestions)
    return render_template('predict.html')

@app.route('/download_report/<prediction>/<probability>', methods=['GET'])
def download_report(prediction, probability):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    suggestions = session.get('suggestions', [])

    # Create PDF
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, "Heart Disease Prediction Report")
    p.setFont("Helvetica", 12)
    p.drawString(100, 730, f"Prediction: {'Positive' if int(prediction) == 1 else 'Negative'}")
    p.drawString(100, 710, f"Probability of Heart Disease: {float(probability):.2f}%")
    p.drawString(100, 690, "Suggestions:")
    y = 670
    for suggestion in suggestions:
        p.drawString(120, y, f"- {suggestion}")
        y -= 20
    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="heart_prediction_report.pdf", mimetype='application/pdf')

# New route for editing profile (to be implemented with a form)
@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        user.name = request.form['name']
        user.email = request.form['email']
        user.age = int(request.form['age'])
        user.gender = request.form['gender']
        db.session.commit()
        flash('Profile updated successfully!')
        return redirect(url_for('profile'))
    return render_template('edit_profile.html', user=user)

if __name__ == '__main__':
    app.run(debug=True)