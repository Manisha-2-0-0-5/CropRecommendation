from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from tamilnaduagroexpert import TamilNaduAgroExpert
from pricefindr import PriceAnalyze
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

# Database Configuration (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farmconnect.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Recommendation Model
class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    district = db.Column(db.String(120), nullable=False)
    month = db.Column(db.String(20), nullable=False)
    irrigation = db.Column(db.Boolean, nullable=False)
    duration = db.Column(db.String(20), nullable=False)
    crops = db.Column(db.String(500), nullable=False)  # JSON string or similar
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Initialize Database
with app.app_context():
    db.create_all()

# Initialize Tamil Nadu Agro Expert and Price Analyzer
data_path = '11.csv'
analyzer = PriceAnalyze(data_path)
expert = TamilNaduAgroExpert()

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match', 'error')
        elif User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
        else:
            new_user = User(username=username, password=password)
            db.session.add(new_user)
            db.session.commit()
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dash.html', username=session['username'])

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if 'username' not in session:
        return redirect(url_for('login'))

    recommendations = None
    error = None

    if request.method == 'POST':
        try:
            district = request.form['district']
            month = request.form['month']
            irrigation = request.form.get('irrigation', 'no') == 'yes'
            duration = request.form.get('duration', 'short')

            # Call the recommend method
            result = expert.recommend(district, month, irrigation, duration)

            # Check if recommendations exist in the result
            if "recommendations" in result:
                recommendations = result["recommendations"]

                # Save recommendations to the database
                crops_str = ', '.join([r['crop'] for r in recommendations])
                new_recommendation = Recommendation(
                    username=session['username'],
                    district=district,
                    month=month,
                    irrigation=irrigation,
                    duration=duration,
                    crops=crops_str
                )
                db.session.add(new_recommendation)
                db.session.commit()
            else:
                error = result.get("error", "No recommendations available.")

        except Exception as e:
            error = str(e)

    return render_template(
        'recommendation.html',
        username=session['username'],
        recommendations=recommendations,
        error=error
    )
@app.route("/price-analysis", methods=["GET", "POST"])
def price_analysis():
    if request.method == "GET":
        return render_template(
            "price_analysis.html",
            districts=analyzer.get_unique_districts(),
            crops=analyzer.get_unique_crops(),
            chart=None,
            recommendations=None,
            selected_crops=[],
            selected_district=None,
            error=None
        )

    try:
        crops = request.form.getlist("crops")
        district = request.form.get("district")

        if not crops or not district:
            raise ValueError("Please select at least one crop and a district.")

        avg_prices = analyzer.analyze_prices_for_district(crops, district)
        chart_filename = analyzer.plot_prices(avg_prices)
        chart_path = f"/static/charts/{chart_filename}"

        recommendations = analyzer.recommend_alternatives(crops, district)

        return render_template(
            "price_analysis.html",
            districts=analyzer.get_unique_districts(),
            crops=analyzer.get_unique_crops(),
            selected_crops=crops,
            selected_district=district,
            chart=chart_path,
            recommendations=recommendations.to_dict(orient="records") if not recommendations.empty else None,
            error=None
        )

    except Exception as e:
        return render_template(
            "price_analysis.html",
            districts=analyzer.get_unique_districts(),
            crops=analyzer.get_unique_crops(),
            selected_crops=crops,
            selected_district=district,
            chart=None,
            recommendations=None,
            error=str(e)
        )
@app.route('/hub')
def hub():
    return render_template('agrihub.html')  # Assuming you have an `agrihub.html` template
if __name__ == '__main__':
    app.run(debug=True)
