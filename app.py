import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import io
import secrets

app = Flask(__name__)
# Generate a random secret key
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Detection Result model
class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    alzheimer_stage = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('results', lazy=True))

# Create database tables
with app.app_context():
    db.create_all()
    
    # Check if admin user exists, if not create one
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin_password = generate_password_hash('admin')
        admin = User(username='admin', name='Administrator', gender='Other', age=0, password=admin_password)
        db.session.add(admin)
        db.session.commit()

# Load the model
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def load_model():
    try:
        # Create the model with the same architecture as in the provided code
        model = timm.create_model('efficientformer_l1', pretrained=False)
        
        # Check if model file exists
        if os.path.exists('best_model_epoch_8.pt'):
            # Load the weights
            state_dict = torch.load('best_model_epoch_8.pt', map_location=torch.device('cpu'), weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            return model
        else:
            # For testing purposes, if model doesn't exist, just return the model without loading weights
            print("Warning: Model file 'best_model_epoch_8.pt' not found. Using untrained model.")
            model.eval()
            return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # For testing purposes, return a dummy model
        print("Using a dummy model for testing.")
        return None

# Initialize model
try:
    model = load_model()
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    model = None

# Image transformation - same as in the provided code
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Helper function to get stage statistics
def get_stage_stats():
    stage_stats = {}
    for stage in class_names:
        count = DetectionResult.query.filter_by(alzheimer_stage=stage).count()
        stage_stats[stage] = count
    return stage_stats

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form.get('username')
    name = request.form.get('name')
    gender = request.form.get('gender')
    age = request.form.get('age')
    password = request.form.get('password')
    
    # Check if username already exists
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash('Username already exists. Please choose a different one.', 'error')
        return redirect(url_for('index'))
    
    # Create new user
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, name=name, gender=gender, age=age, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    flash('Account created successfully! Please login.', 'success')
    return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Check if it's the admin
    if username == 'admin':
        user = User.query.filter_by(username='admin').first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['name'] = user.name
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials. Please try again.', 'error')
            return redirect(url_for('index'))
    
    # Regular user login
    user = User.query.filter_by(username=username).first()
    
    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id
        session['username'] = user.username
        session['name'] = user.name
        session['gender'] = user.gender
        session['age'] = user.age
        session['is_admin'] = False
        return redirect(url_for('dashboard'))
    
    flash('Invalid username or password. Please try again.', 'error')
    return redirect(url_for('index'))

@app.route('/reset_password', methods=['POST'])
def reset_password():
    username = request.form.get('reset_username')
    new_password = request.form.get('new_password')
    
    user = User.query.filter_by(username=username).first()
    
    if not user:
        flash('Username not found. Please try again.', 'error')
        return redirect(url_for('index'))
    
    user.password = generate_password_hash(new_password)
    db.session.commit()
    
    flash('Password reset successfully! Please login with your new password.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for('index'))
    
    if session.get('is_admin', False):
        return redirect(url_for('admin_dashboard'))
    
    # Get user's previous results
    user_results = DetectionResult.query.filter_by(user_id=session['user_id']).order_by(DetectionResult.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html', 
                          username=session['username'], 
                          name=session['name'],
                          results=user_results)

@app.route('/admin')
def admin_dashboard():
    if 'user_id' not in session or not session.get('is_admin', False):
        flash('You do not have permission to access the admin panel.', 'error')
        return redirect(url_for('index'))
    
    # Get all users except admin
    users = User.query.filter(User.username != 'admin').order_by(User.created_at.desc()).all()
    
    # Get recent detection results
    recent_results = DetectionResult.query.order_by(DetectionResult.created_at.desc()).limit(10).all()
    
    # Count users by Alzheimer's stage
    stage_stats = get_stage_stats()
    
    return render_template('admin.html', 
                          users=users, 
                          results=recent_results,
                          stats=stage_stats,
                          total_users=len(users),
                          total_detections=DetectionResult.query.count())

@app.route('/detect')
def detect():
    if 'user_id' not in session:
        flash('Please login to access the detection page.', 'error')
        return redirect(url_for('index'))
    
    # Check if the user is admin
    if session.get('is_admin', False):
        flash('Admin cannot perform Alzheimer\'s disease detection.', 'error')
        return redirect(url_for('admin_dashboard'))
    
    # Count users by Alzheimer's stage for statistics
    stage_stats = get_stage_stats()
    
    return render_template('detect.html', 
                          username=session.get('username'),
                          name=session.get('name'),
                          gender=session.get('gender'),
                          age=session.get('age'),
                          stats=stage_stats)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'user_id' not in session:
        flash('Please login to perform detection.', 'error')
        return redirect(url_for('index'))
    
    if 'image' not in request.files:
        flash('No image uploaded', 'error')
        return redirect(url_for('detect'))
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No image selected', 'error')
        return redirect(url_for('detect'))
    
    try:
        # Process the image directly in memory without saving
        image = Image.open(file).convert('RGB')
        input_image = test_transform(image).unsqueeze(0)
        
        # Check if model is loaded
        if model is None:
            # For testing/demo purposes, return a random prediction
            import random
            predicted_class_name = random.choice(class_names)
            confidence = random.uniform(0.7, 0.99)
            flash('Note: Using random prediction as model is not available.', 'warning')
        else:
            # Using the approach from the provided code
            with torch.no_grad():
                output = model(input_image)
            
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = F.softmax(output, dim=1)[0][predicted_class].item()
            predicted_class_name = class_names[predicted_class]
        
        # Save results to session
        session['result'] = {
            'alzheimer_stage': predicted_class_name,
            'confidence': confidence
        }
        
        # Save result to database
        new_result = DetectionResult(
            user_id=session['user_id'],
            alzheimer_stage=predicted_class_name,
            confidence=confidence
        )
        db.session.add(new_result)
        db.session.commit()
        
        # Get stage statistics for the template
        stage_stats = get_stage_stats()
        
        return render_template('detect.html', 
                              result=session['result'],
                              username=session['username'],
                              name=session['name'],
                              gender=session['gender'],
                              age=session['age'],
                              stats=stage_stats)
    
    except Exception as e:
        flash(f'Error processing image: {str(e)}', 'error')
        return redirect(url_for('detect'))

@app.route('/download_result')
def download_result():
    if 'user_id' not in session or 'result' not in session:
        flash('No detection result available to download.', 'error')
        return redirect(url_for('detect'))
    
    result_text = f"User Information:\n"
    result_text += f"Name: {session['name']}\n"
    result_text += f"Username: {session['username']}\n"
    result_text += f"Age: {session['age']}\n"
    result_text += f"Gender: {session['gender']}\n\n"
    result_text += f"Alzheimer's Disease Detection Result:\n"
    result_text += f"Stage: {session['result']['alzheimer_stage']}\n"
    result_text += f"Confidence: {session['result']['confidence']:.2f}\n"
    result_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add recommendations based on the stage
    result_text += "Recommendations:\n"
    stage = session['result']['alzheimer_stage']
    
    if stage == "NonDemented":
        result_text += "- Continue regular cognitive health check-ups\n"
        result_text += "- Maintain a healthy lifestyle with regular exercise\n"
        result_text += "- Engage in mentally stimulating activities\n"
    elif stage == "VeryMildDemented":
        result_text += "- Consult with a neurologist for further evaluation\n"
        result_text += "- Consider cognitive enhancement exercises\n"
        result_text += "- Establish regular routines to maintain independence\n"
    elif stage == "MildDemented":
        result_text += "- Seek specialized medical care from a neurologist\n"
        result_text += "- Consider medication options to slow progression\n"
        result_text += "- Implement memory aids and safety measures at home\n"
    elif stage == "ModerateDemented":
        result_text += "- Immediate consultation with a specialist is recommended\n"
        result_text += "- Discuss treatment plans and care options\n"
        result_text += "- Consider support groups for both patient and caregivers\n"
    
    result_text += "\nDisclaimer: This is an AI-assisted detection and should not replace professional medical diagnosis."
    
    buffer = io.BytesIO()
    buffer.write(result_text.encode('utf-8'))
    buffer.seek(0)
    
    return send_file(buffer, 
                    as_attachment=True, 
                    download_name=f"alzheimers_detection_{session['username']}.txt", 
                    mimetype='text/plain')

@app.route('/user/<int:user_id>')
def user_details(user_id):
    if 'user_id' not in session or not session.get('is_admin', False):
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('index'))
    
    user = User.query.get_or_404(user_id)
    results = DetectionResult.query.filter_by(user_id=user_id).order_by(DetectionResult.created_at.desc()).all()
    
    return render_template('user_details.html', user=user, results=results)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'user_id' not in session or not session.get('is_admin', False):
        flash('You do not have permission to perform this action.', 'error')
        return redirect(url_for('index'))
    
    user = User.query.get_or_404(user_id)
    
    # Delete all user's results first
    DetectionResult.query.filter_by(user_id=user_id).delete()
    
    # Delete the user
    db.session.delete(user)
    db.session.commit()
    
    flash(f'User {user.username} has been deleted.', 'success')
    return redirect(url_for('admin_dashboard'))

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code=404, error_message="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_code=500, error_message="Internal server error"), 500

if __name__ == '__main__':
    app.run(debug=True)

