# Public Sentiment Toward ChatGPT: Classification and Forecasting

How do users feel about ChatGPT, and is that changing over time? This project classifies the sentiment of ChatGPT app reviews and forecasts how the share of negative reviews will develop.

## Dataset

- About **1.03 million** ChatGPT app reviews with star ratings and dates
- A random sample of **50,000** reviews was used to keep processing fast; **47,453** remained after cleaning
- Period covered: **July 2023 – May 2026**
- Source: [add dataset link here]

## Approach

1. **Cleaning:** lowercasing, removing URLs and non-letters, stop-word removal, lemmatisation (NLTK)
2. **Labelling:** star ratings turned into sentiment (1–2 = negative, 3 = neutral, 4–5 = positive)
3. **Exploratory analysis:** rating and sentiment distributions, top TF-IDF terms per sentiment
4. **Balancing:** the data was heavily skewed (87.5% positive, 8.5% negative, 4.1% neutral), so each class was downsampled to 1,923 reviews
5. **Classification:** TF-IDF features (unigrams and bigrams, 5,000 features) with Logistic Regression, 80/20 train-test split
6. **Forecasting:** monthly share of negative reviews modelled with ARIMA(1,1,1), forecast six months ahead

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.61 | 0.63 | 0.62 |
| Neutral | 0.44 | 0.46 | 0.45 |
| Positive | 0.59 | 0.53 | 0.56 |
| **Overall accuracy** | | | **0.54** |

- **Accuracy of 54%** across three balanced classes, compared with 33% for random guessing
- **Neutral reviews were hardest to classify**, as expected: they often mix praise and complaints
- **Forecast:** the negative share is predicted to ease slightly, from about 9.1% to 8.6% between June and November 2026

## Key finding: star ratings are noisy labels

Many 1- and 2-star reviews contain positive words such as "very good" or "good". In other words, users often give a rating that doesn't match their text. This label noise limits how well any model can learn sentiment from ratings, and is a main reason accuracy stays moderate.

## Limitations and next steps

- Labels come from star ratings, not human annotation
- Only English text is kept after cleaning, so non-English reviews lose most of their content
- Downsampling discards most of the data; class weights or a transformer model (e.g. BERT) could improve results
- ARIMA uses only past negative share and does not account for events such as model releases

## Tech stack

Python · pandas · NumPy · NLTK · scikit-learn · statsmodels · Matplotlib · seaborn · Jupyter

## How to run

1. Install the dependencies:
   ```bash
   pip install pandas numpy nltk scikit-learn statsmodels matplotlib seaborn jupyter
   ```
2. Download `chatgpt_reviews.csv` from the source above and put it in the project folder.
3. Open `Business project in big data.ipynb` and run all cells.
