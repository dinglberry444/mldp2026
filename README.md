# COE Premium Predictor (mldp2026)

Machine Learning for Developers (CAI2C08) — end-of-semester project.
Diploma in Applied Artificial Intelligence, Temasek Polytechnic.

## Project summary

A B2B regression tool that predicts the Singapore **COE premium** before the
bidding result is announced, so used-car dealers can price vehicles accurately
and protect their margins. Built following the CRISP-DM pipeline.

- **Task:** Supervised regression
- **Target:** `premium` (SGD)
- **Best model:** Tuned Random Forest Regressor (RMSE ≈ $5,300, R² ≈ 0.98)

## Files

| File | Description |
|------|-------------|
| `MLDP Program Codes Submission Template.ipynb` | Main notebook (business understanding → EDA → cleaning → modelling → evaluation → tuning) |
| `app.py` | Streamlit web app for live premium predictions |
| `COEBiddingResultsPrices.csv` | Dataset (COE bidding results, data.gov.sg) |
| `best_coe_model.pkl` | Saved trained model |
| `model_columns.pkl` | Feature column order used by the app |
| `requirements.txt` | Python dependencies |

## How to run

Install dependencies:

```
pip install -r requirements.txt
```

Run the notebook in VS Code (Jupyter), or launch the web app:

```
streamlit run app.py
```

The app opens at http://localhost:8501.
