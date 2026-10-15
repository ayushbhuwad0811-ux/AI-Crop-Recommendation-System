# Crop Recommendation System Using Machine Learning

## Project overview
This project trains a Random Forest classifier to predict a crop label from seven input features: nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, soil pH, and rainfall.

**Important:** This is an educational prototype, not a substitute for local agronomic guidance. Dataset units, provenance, and regional suitability must be verified before practical use.

## 1. Dataset
Obtain a crop recommendation CSV from a reputable educational/open-data source, and save it as:

`data/crop_data.csv`

Required columns (case-insensitive):
- `N`
- `P`
- `K`
- `temperature`
- `humidity`
- `ph`
- `rainfall`
- `label` (crop name/class)

A commonly used public benchmark is titled **Crop Recommendation Dataset** and is hosted on Kaggle. Review its licence and source notes before redistribution. The dataset is not bundled here so its provenance and licence are not misrepresented.

## 2. Setup (Windows)
Open Command Prompt in this folder and run:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py` is not available, try `python` instead.

## 3. Train and evaluate
```bash
python train_model.py
```
The script creates:
- `models/crop_model.joblib`
- `reports/evaluation.json`

The test set is held out from training. Report the measured test accuracy from your run; do not claim a result before running the code.

## 4. Run the web app
```bash
streamlit run app.py
```
Streamlit opens a local browser page. Enter feature values and select **Recommend crop**.

## 5. ML pipeline
1. Load CSV dataset
2. Validate required columns and convert input features to numeric
3. Remove rows with missing required values
4. Split into training (80%) and test (20%) sets
5. Train a Random Forest classifier
6. Evaluate using held-out accuracy and a classification report
7. Save the trained model
8. Load model in Streamlit and predict a crop class

## 6. Suggested report sections
1. Title page
2. Abstract
3. Introduction and problem statement
4. Objectives and SDG 2 connection
5. Dataset source, licence, columns, and limitations
6. Methodology / ML pipeline
7. Model selection and train/test split
8. Results (insert your actual measured results)
9. Application screenshots
10. Limitations, future scope, conclusion
11. References

## 7. Limitations and future scope
- Prediction quality depends on dataset quality and regional representativeness.
- The model does not account for every local factor such as soil texture, irrigation access, season, pests, seed variety, market demand, or current forecasts.
- Future work: compare Decision Tree / Random Forest / KNN, add feature-importance plots, support region-specific datasets, and validate with agricultural experts.

## Project structure
```text
Crop_Recommendation_ML/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── crop_data.csv        # add dataset here
├── models/                  # generated after training
└── reports/                 # generated after training
```
