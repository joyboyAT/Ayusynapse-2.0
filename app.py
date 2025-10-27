from flask import Flask, render_template, request, jsonify
from clinical_trials_client import ClinicalTrialsClient
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create and configure the Flask application
app = Flask(__name__)
client = ClinicalTrialsClient(cache_enabled=True)

# Ensure template and static directories exist
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

@app.route('/')
def home():
    """Homepage with search form"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clinical Trials Search</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .search-form {
                background-color: #fff;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input, select {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                cursor: pointer;
                border-radius: 4px;
                font-size: 16px;
            }
            button:hover {
                background-color: #2980b9;
            }
            #results {
                margin-top: 20px;
            }
            .trial-card {
                background-color: #fff;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .trial-title {
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 5px;
            }
            .trial-status {
                display: inline-block;
                padding: 3px 8px;
                background-color: #2ecc71;
                color: white;
                border-radius: 3px;
                font-size: 12px;
            }
            .trial-details {
                margin-top: 10px;
            }
            .loading {
                text-align: center;
                padding: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Clinical Trials Search</h1>
            
            <div class="search-form">
                <div class="form-group">
                    <label for="condition">Medical Condition:</label>
                    <input type="text" id="condition" placeholder="e.g., diabetes, cancer, heart disease" required>
                </div>
                
                <div class="form-group">
                    <label for="status">Trial Status:</label>
                    <select id="status">
                        <option value="RECRUITING">Recruiting</option>
                        <option value="NOT_YET_RECRUITING">Not Yet Recruiting</option>
                        <option value="ACTIVE_NOT_RECRUITING">Active, not recruiting</option>
                        <option value="COMPLETED">Completed</option>
                        <option value="TERMINATED">Terminated</option>
                        <option value="WITHDRAWN">Withdrawn</option>