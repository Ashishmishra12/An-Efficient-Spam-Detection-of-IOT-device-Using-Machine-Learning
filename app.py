import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
from sklearn.metrics import confusion_matrix

app = Flask(__name__)  # Initialize the Flask App

# Load the model
model = pickle.load(open('spam.pkl', 'rb'))

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('time', inplace=True)
        return render_template("preview.html", df_view=df)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_feature = [x for x in request.form.values()]
    final_features = [np.array(int_feature)]
    result = model.predict(final_features)
    
    if result == 1:
        result = "Spam detected"
    else:
        result = "No spam"
    
    return render_template('prediction.html', prediction_text=result)

@app.route('/performance')
def performance():
    generate_conf_matrix()  # Ensure Confusion Matrix is generated before rendering
    return render_template('performance.html')

# ------------------------- CONFUSION MATRIX GENERATION -------------------------
def generate_conf_matrix():
    """Generates and saves the confusion matrix as an image."""
    save_path = "model/static/conf_matrix.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Dummy test data for confusion matrix (Replace with actual test set if available)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)

    cm = confusion_matrix(y_true, y_pred)

    # Plot the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Spam", "Spam"], yticklabels=["No Spam", "Spam"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Save the image
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved at: {save_path}")

# ------------------------- RUN APP -------------------------
if __name__ == "__main__":
    app.run(debug=True)
