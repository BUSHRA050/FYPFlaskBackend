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

nltk.download('stopwords')

app = Flask(__name__)
CORS(app) 


atlas_username = os.getenv("ATLAS_USERNAME")
atlas_password = os.getenv("ATLAS_PASSWORD")
atlas_cluster_uri = os.getenv("ATLAS_CLUSTER_URI")
database_name = os.getenv("DATABASE_NAME")
jobs_collection_name = os.getenv("JOB_COLLECTION_NAME")
resume_collection_name = os.getenv("RESUME_COLLECTION_NAME")

uri = f"mongodb+srv://{atlas_username}:{atlas_password}@{atlas_cluster_uri}/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("Mongo Db Error: ",e)

client = MongoClient(uri)
db = client[database_name]

# Define job_collection and resume_collection
job_collection = db[jobs_collection_name]
resume_collection = db[resume_collection_name]

# # Print data in job_collection
# print("Data in job_collection:")
# for job_document in job_collection.find():
#     print(job_document)

# # Print data in resume_collection
# print("\nData in resume_collection:")
# for resume_document in resume_collection.find():
#     print(resume_document)

# ====>     PREPROCESSING TEXT <====
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]','',text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# ====>     RESUME ANALYZER <====
def analyze_resume(job_description, resume):
    vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
    job_vec = vectorizer.fit_transform([job_description])
    resume_vec = vectorizer.transform([resume])

    cos_sim = cosine_similarity(job_vec, resume_vec)[0][0]
    cos_sim_percent = cos_sim * 100
    
    return cos_sim_percent


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

    # Add relevant fields from the resume data to the resume_text
    resume_text += resume_data.get("name", "") + "\n"
    resume_text += resume_data.get("location", "") + "\n"
    resume_text += resume_data.get("phone", "") + "\n"
    resume_text += resume_data.get("email", "") + "\n"
    resume_text += resume_data.get("jobDescription", "") + "\n"
    resume_text += resume_data.get("objective", "") + "\n"
    resume_text += resume_data.get("about", "") + "\n"

    cos_sim_percent = analyze_resume(job_description, resume_text)

    if cos_sim_percent > 40:
        eligibility = "Hooray! You're Eligible to Apply"
    else:
        eligibility = "Sorry! You're not Eligible for this Job"
    
    return jsonify({'Cosine_Similarity': cos_sim_percent, 'Eligibility': eligibility}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
