from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Full path to your model file (make sure iris_model.pkl is in the same folder as app.py)
model_path = os.path.join(os.path.dirname(__file__), 'iris_model.pkl')

print(f"Looking for model at: {model_path}")
print("Model file exists:", os.path.exists(model_path))

# Load the trained model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Mapping numeric labels to species names
label_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# Image URLs for each species
iris_images = {
    "Setosa": "https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg",
    "Versicolor": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTynH7icam0GALTs77dufBndHsZcOwdloh2-A&s",
    "Virginica": "https://www.fs.usda.gov/wildflowers/beauty/iris/Blue_Flag/images/iris_virginica/iris_virginica_virginica_lg.jpg"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form input
        features = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])
        ]
        
        # Predict numeric label
        pred_num = model.predict([features])[0]

        # Map to species name
        prediction = label_map.get(pred_num, "Unknown")

        # Get image URL
        image_url = iris_images.get(prediction, "")

        # Render result page with prediction and image
        return render_template('result.html', prediction=prediction, image_url=image_url)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
