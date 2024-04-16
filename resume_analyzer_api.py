from flask import Flask, request, jsonify
from flask_cors import CORS 
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import os
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import numpy as np
import sys
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)
CORS(app) 

jobs_collection_name ="Jobs"
resume_collection_name = "Resume"
uri = f"mongodb+srv://bhsjobportal:fbz4lRVJYtXs7qKe@cluster0.itkhalq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("Mongo Db Error: ", e)

client = MongoClient(uri)
db = client["test"]

job_collection = db[jobs_collection_name]
resume_collection = db[resume_collection_name]

# Initialize the Porter Stemmer
stemmer = PorterStemmer()
# ====>     PREPROCESSING TEXT <====
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]  
    preprocessed_text = ' '.join(set(tokens))  
    print(preprocessed_text)
    sys.stdout.flush()
    return preprocessed_text
   
def analyze_resume(job_description, resume):
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    job_vec = vectorizer.fit_transform([job_description])
    resume_vec = vectorizer.transform([resume])
    cos_sim = cosine_similarity(job_vec, resume_vec)[0][0]
    cos_sim_percent = cos_sim * 100  
    return cos_sim_percent

def jaccard_similarity(job_description, resume):
    preprocessed_job_description = preprocess_text(job_description)
    preprocessed_resume = preprocess_text(resume)
    query_words = set(preprocessed_job_description.split())
    document_words = set(preprocessed_resume.split())
    intersection = len(query_words.intersection(document_words))
    union = len(query_words.union(document_words))
    return intersection / union * 100

def minkowski_distance(job_description, resume, p=2):
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    job_vec = vectorizer.fit_transform([job_description])
    resume_vec = vectorizer.transform([resume])
    minkowski_dist = np.linalg.norm((job_vec - resume_vec).toarray(), ord=p)
    minkowski_sim = 1 / (1 + minkowski_dist)
    return minkowski_sim * 100

@app.route('/analyze_resume_and_job', methods=['POST'])
def analyze_resume_and_job_api():
    if 'job_id' not in request.json or 'user_id' not in request.json:
        return jsonify({'Error': 'Job ID or User ID is missing'})
 # Convert job_id string to ObjectId
    job_id = ObjectId(request.json['job_id'])
    user_id = request.json['user_id']

    job_description_data = job_collection.find_one({"_id": job_id})
    resume_data = resume_collection.find_one({"userId": user_id})

    if not job_description_data or not resume_data:
        return jsonify({'Error': 'Job description or Resume data not found in MongoDB'})

    job_description = job_description_data.get("description", "")
    resume_text = ""
    resume_text += resume_data.get("jobDescription", "") + "\n"
    resume_text += resume_data.get("about", "") + "\n"

    cos_sim_percent = analyze_resume(job_description, resume_text)
    jaccard_sim_percent = jaccard_similarity(job_description, resume_text)
    minkowski_sim_percent = minkowski_distance(job_description, resume_text)

    similarity_scores = {
        'Cosine Similarity': cos_sim_percent,
        'Jaccard Similarity': jaccard_sim_percent,
        'Minkowski Distance': minkowski_sim_percent
    }

    print("Cosine Similarity Score:", cos_sim_percent)
    sys.stdout.flush()

    print("Jaccard Similarity Score:", jaccard_sim_percent)
    sys.stdout.flush()

    print("Minkowski Distance Score:", minkowski_sim_percent)
    sys.stdout.flush()

    best_algo = max(similarity_scores, key=similarity_scores.get)
    best_score = similarity_scores[best_algo]

    if best_score >= 50:
        eligibility = "Hooray! You're Eligible to Apply"
    else:
        eligibility = "Sorry! You're not Eligible for this Job"
    
    return jsonify({'Best_Score_Algo': best_algo, 'Score': best_score, 'Eligibility': eligibility}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
