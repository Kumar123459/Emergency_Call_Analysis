# Emergency Call Analysis

A Streamlit decision-support prototype that analyzes an emergency call transcript for a likely incident topic, caller sentiment, model confidence, severity, and suggested assistance.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start the app from this directory:

   ```powershell
   streamlit run app.py
   ```

The app trains and saves a model from `dataset/emergency_sentiment.csv` the first time it runs if no model is present. To train it separately, run:

```powershell
python train.py
```

## Important

This is a small-dataset prototype. Its predictions and suggested responses may be inaccurate and must not replace emergency services or a trained dispatcher. For an active emergency, contact your local emergency number.
