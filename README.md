# 🕵️‍♂️ Fake Job Posting Classification (Real or Fake)

This project builds and deploys a machine learning model to classify job postings as real or fake using a supervised text classification approach.

---

## Dataset

- Source: [Kaggle - Real or Fake Job Posting](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Target variable: `fraudulent` (0 = Real, 1 = Fake)

---

## Project Features

- Exploratory Data Analysis (EDA)
- Text preprocessing (cleaning, tokenization, lemmatization)
- TF-IDF vectorization
- Supervised ML models: Logistic Regression, Naive Bayes, Random Forest
- Hyperparameter tuning with GridSearchCV
- Streamlit app for live predictions

---

## Requirements

> NLP library necessary fact to do.
```bash
pip install nltk
nltk.download()
```
> To install remaining libraries
```bash
pip install pandas matplotlib seaborn wordcloud scikit-learn joblib streamlit
```
## Final Model Step
Using joblib, <b>best_model.pkl</b> and <b>vectorizer.pkl</b> will get.

## To Deploy or Run at Website by using streamlit

```bash
streamlit run FakeJobClassification_App.py
```
