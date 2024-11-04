from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from itsdangerous import URLSafeTimedSerializer

app = Flask(__name__)

# Configure MySQL database connection
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+mysqlconnector://root:Saahil123@localhost:3306/Project_final"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = "please-remember-to-change-me"
app.secret_key = "your_secret_key"

reset_serializer = URLSafeTimedSerializer(app.secret_key)

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app,resources={r"/*": {"origins": "http://localhost:3000"}},supports_credentials=True)

# Define User model
class User(db.Model):
    __tablename__ = "User_Credentials"
    SRN = db.Column(db.String(50), nullable=False, unique=True, primary_key=True)
    password_hash = db.Column(db.String(255), nullable=False)
    password_reset_token = db.Column(db.String(255), nullable=True)

# Initialize NLP tools
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load datasets
def load_courses_df():
    query = "SELECT * FROM courses_final"
    with app.app_context():
        return pd.read_sql(query, db.engine)

def load_job_df():
    return pd.read_csv('job_data_filtered.csv')

# Create the database tables and load initial data
with app.app_context():
    db.create_all()
    courses_df = load_courses_df()
    job_df = load_job_df()

# Define text preprocessing function
def stem_text(text):
    if isinstance(text, str):
        words = nltk.word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(stemmer.stem(word) for word in filtered_words)
    return ""

# Process job descriptions
job_df['job_description_stemmed'] = job_df['job_description'].apply(stem_text)

# Initialize vectorizers
def initialize_vectorizers():
    global tfidf_vectorizer_courses, tfidf_matrix_courses, tfidf_vectorizer_jobs, tfidf_matrix_jobs
    
    # For courses
    courses_df['combined_text'] = courses_df['course_description_stemmed'].astype(str) + ' ' + courses_df['course_skills'].astype(str)
    tfidf_vectorizer_courses = TfidfVectorizer(stop_words='english')
    tfidf_matrix_courses = tfidf_vectorizer_courses.fit_transform(courses_df['combined_text'].fillna(''))
    
    # For jobs
    tfidf_vectorizer_jobs = TfidfVectorizer(stop_words='english')
    tfidf_matrix_jobs = tfidf_vectorizer_jobs.fit_transform(job_df['job_description_stemmed'].fillna(''))

# Initialize vectorizers within app context
with app.app_context():
    initialize_vectorizers()

# Define recommendation function
def recommend_items(student_input, tfidf_matrix, vectorizer, data_df):
    # Split and filter stop words from the input
    student_input = student_input.split(",")
    filtered_input = [word.strip() for word in student_input if word.strip() and word not in stop_words]

    # Handle case where only stop words or no input is provided
    if not filtered_input:
        raise ValueError("Input only contains stop words or is empty. Please provide a more detailed input.")
    
    # Vectorize the filtered input
    student_input_vector = vectorizer.transform([" ".join(filtered_input)])

    # Calculate similarity scores and get top recommendations
    similarity_scores = cosine_similarity(student_input_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]
    return data_df.iloc[top_indices]

@app.route('/delete-user', methods=['DELETE'])
@jwt_required()
def delete_user():
    current_user_srn = get_jwt_identity()
    user = User.query.filter_by(SRN=current_user_srn).first()
    
    if not user:
        return jsonify({"msg": "User not found"}), 404
    
    try:
        db.session.delete(user)
        db.session.commit()
        return jsonify({"msg": "User deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"msg": "Error deleting user", "error": str(e)}), 500 #transaction

@app.route('/signup', methods=['POST'])
def signup():
    srn_r = request.json.get("SRN")
    password = request.json.get("password")

    if User.query.filter_by(SRN=srn_r).first():
        return jsonify({"msg": "User already exists"}), 400 #trigger

    hashed_password = generate_password_hash(password)
    new_user = User(SRN=srn_r, password_hash=hashed_password)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"msg": "User created successfully"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"msg": "Error creating user", "error": str(e)}), 500 #transaction

@app.route('/login', methods=['POST'])
def login():
    srn_r = request.json.get("SRN")
    password = request.json.get("password")

    user = User.query.filter_by(SRN=srn_r).first()
    if not user or not check_password_hash(user.password_hash, password):
        return {"msg": "Wrong username or password"}, 401

    access_token = create_access_token(identity=srn_r)
    response = {"access_token": access_token}
    return response

@app.route('/recommend/<type>', methods=['POST'])
@jwt_required()
def recommend(type):
    current_user = get_jwt_identity()
    data = request.get_json()
    student_input = data.get('student_input')
    
    if not student_input:
        return jsonify({'error': 'No student input provided'}), 400

    try:
        if type == 'courses':
            # Refresh course data
            global courses_df, tfidf_matrix_courses, tfidf_vectorizer_courses
            courses_df = load_courses_df()
            initialize_vectorizers()
            recommended_items = recommend_items(student_input, tfidf_matrix_courses, tfidf_vectorizer_courses, courses_df)
            return jsonify(recommended_items[['course_title', 'course_url']].to_dict(orient='records'))
        elif type == 'jobs':
            recommended_items = recommend_items(student_input, tfidf_matrix_jobs, tfidf_vectorizer_jobs, job_df)
            return jsonify(recommended_items[['job_title']].to_dict(orient='records'))
        else:
            return jsonify({'error': 'Invalid recommendation type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/request-password-reset', methods=['POST'])
def request_password_reset():
    srn_r = request.json.get("SRN")
    user = User.query.filter_by(SRN=srn_r).first()

    if not user:
        return jsonify({"msg": "User not found"}), 404

    token = reset_serializer.dumps(user.SRN, salt='password-reset-salt')
    user.password_reset_token = token
    db.session.commit()
    
    return jsonify({"msg": "Password reset token generated", "token": token}), 200

@app.route('/reset-password', methods=['POST'])
def reset_password():
    token = request.json.get("token")
    new_password = request.json.get("new_password")

    try:
        srn_r = reset_serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except Exception as e:
        return jsonify({"msg": "Invalid or expired token"}), 400

    user = User.query.filter_by(SRN=srn_r, password_reset_token=token).first()

    if not user:
        return jsonify({"msg": "Invalid token or user not found"}), 404

    user.password_hash = generate_password_hash(new_password)
    user.password_reset_token = None
    db.session.commit()

    return jsonify({"msg": "Password reset successful"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)