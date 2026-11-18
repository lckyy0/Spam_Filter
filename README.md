# Naive Bayes Spam Filter

A spam classifier built from scratch with numpy and pandas, without scikit-learn or any other ML library.
It is trained on the Enron-Spam corpus (about 30,500 labelled emails after cleaning) and 
reaches 98.52% accuracy and 99.01% precision on a held-out test set.

**Contents**
[Results](#results) · [Dataset](#dataset) · [Pipeline](#pipeline) · [Design choices](#design-choices) · [Evaluation](#evaluation) · [Reproducing the results](#reproducing-the-results) · [Repository layout](#repository-layout)  · [Data credits](#data-credits)

## Results

Test set: 6,099 emails the model never saw during training (3,140 ham, 2,959 spam).

| Metric | Value |
|---|---|
| Accuracy | 98.52% |
| Precision (spam) | 99.01% |
| Recall (spam) | 97.94% |
| F1 (spam) | 98.47% |
| Specificity (ham recognised as ham) | 99.08% |
| Matthews correlation coefficient | 0.9705 |
| Ham sent to spam (false positives) | 29 (0.92% of ham) |
| Spam let through (false negatives) | 61 (2.06% of spam) |

|  | predicted ham | predicted spam |
|---|---|---|
| **actual ham** | 3,111 | 29 |
| **actual spam** | 61 | 2,898 |

A classifier that always answered "ham" would score 51.48% on this test set, so the class balance alone does not explain the result. See [Evaluation](#evaluation) for how far these numbers can be trusted.

## Dataset

The data is the Enron-Spam corpus of Metsis, Androutsopoulos and Paliouras (see [Data credits](#data-credits)). Its ham messages come from the mailboxes of Enron employees, and the spam was assembled separately from them.

The single-file CSV version used here has these properties:

| Property | Value |
|---|---|
| Rows | 33,716 (17,171 spam, 16,545 ham) |
| Columns used | `Subject`, `Message`, `Spam/Ham` (`Message ID` and `Date` are ignored) |
| Date range | 1999-12-10 to 2005-09-06 |
| Empty values | 289 subjects and 371 message bodies |
| Exact duplicate texts | 3,222 (2,587 spam, 635 ham) |
| Usable rows after deduplication | 30,494 (15,910 ham, 14,584 spam; 52.2% / 47.8%) |
| Text length (subject + body) | mean 1,505 characters, median 711, 99th percentile 9,895, maximum 228,368 |

The CSV is downloaded when the notebook first runs and is not stored in this repository.

## Pipeline

Everything below is implemented in `spam_filter.ipynb`, in this order.

### 1. Loading and validation

The subject and body are joined into one string (`Subject + " " + Message`), with empty values replaced by empty strings. The label becomes 1 for `spam` and 0 for `ham`.

Before anything else, the loader counts distinct spam texts. If fewer than half of the spam rows are unique, it raises a `ValueError`. A healthy copy has about 85% unique spam texts.

### 2. Deduplication, shuffling and splitting

Exact duplicate texts are dropped, keeping the first occurrence, so that an email cannot sit in both the training and the test set. The remaining emails are shuffled with `random_state=42`, then split sequentially: the first 80% for training (24,395 emails, 47.7% spam) and the last 20% for testing (6,099 emails, 48.5% spam). There is no separate validation set because nothing is tuned on the data (see the next section).

Only exact duplicates are removed. Near-duplicates, such as spam campaigns that vary a few words, stay in and can land on both sides of the split.

### 3. Text preprocessing

1. Lowercase the text.
2. Extract tokens with the regular expression `[a-z0-9]+`. This keeps letters and digits and discards all punctuation, so `don't` becomes `don` and `t`.
3. Drop tokens shorter than two characters.
4. Drop English stopwords from NLTK's list (198 words in the NLTK data version used for development).

There is no stemming, lemmatisation or n-gram extraction. Numbers are kept as tokens. The training vocabulary has 140,197 distinct tokens.

### 4. Model

The classifier is a Naive Bayes model in log space. For an email with tokens $w_1, \dots, w_m$ the score is

$$
\text{score} = \log\frac{P(\text{spam})}{P(\text{ham})} + \sum_{i=1}^{m} \log\frac{P(w_i \mid \text{spam})}{P(w_i \mid \text{ham})}
$$

and a positive score means spam. The word probabilities are document frequencies with add-one smoothing:

$$
P(w \mid c) = \frac{n_{w,c} + 1}{N_c + 2}
$$

where $n_{w,c}$ is the number of training emails of class $c$ that contain $w$ at least once, and $N_c$ is the number of training emails of that class. The extra 2 in the denominator is the usual correction for a binary present/absent feature.

The class prior comes from the training split: $\log(11{,}625 / 12{,}770) \approx -0.094$, a very slight lean toward ham.

Two details are worth knowing:

- Words are counted once per email when training, but at scoring time every occurrence of a word contributes. The model is therefore a hybrid of Bernoulli-style estimation and multinomial-style scoring, not a strict implementation of either.
- Words that are absent from an email contribute nothing to its score, as in the multinomial model. A full Bernoulli model would also penalise their absence.

Scores are summed in log space because a product of a few hundred per-word probabilities, each well below 1, falls past the range of a 64-bit float and becomes exactly 0 for long emails.

### 5. Training

Training is a single pass over the training emails. For each email the set of its tokens is counted into a spam table or a ham table, and the log-odds of every word are computed once, in vectorised pandas operations, and stored in a dictionary. There is no optimisation loop, learning rate or epoch count. The result is exact and deterministic.

### 6. Inference

An email is tokenised the same way as the training data. Its score is the prior plus the sum of the stored log-odds of its tokens, and any token the model has never seen is ignored (it adds 0). The decision threshold is a score of 0.

`SpamFilter.predict_proba` returns a spam probability by applying the logistic function to the score, clipped to $[-500, 500]$ to keep the exponential finite. Naive Bayes scores are typically overconfident, because the independence assumption counts correlated words as separate evidence. The probabilities therefore saturate close to 0 or 1 and should be read as a ranking of confidence, not as calibrated probabilities.

The model is exposed as a small class with `fit`, `score`, `predict` and `predict_proba` methods.

## Design choices

All values below were fixed in advance, and no setting was changed after looking at the test results.

| Setting | Value | Reasoning |
|---|---|---|
| Train / test split | 80 / 20, sequential after a shuffle | Standard, and the test set is large (6,099 emails) |
| Random seed | 42 | Makes the shuffle, and so every number here, reproducible |
| Smoothing | add-one, $(n+1)/(N+2)$ | Keeps every word probability away from 0 and 1 |
| Token pattern | `[a-z0-9]+` | Simple, fast, and needs no tokenizer data download |
| Minimum token length | 2 characters | Drops single letters and stray digits |
| Stopwords | NLTK English list | Removes high-frequency words that carry little signal |
| Decision threshold | score > 0 (probability > 0.5) | The Bayes decision rule when both error types cost the same |
| Chart filter | words in at least 100 emails, top 15 | Display only; it does not affect the model |


## Evaluation

**Protocol.** The model was trained on the training split and evaluated on the test split. No cross-validation was run on Enron-Spam, because the model has no tuned parameters to select.

**Metrics.** Spam is the positive class. Precision is the fraction of emails flagged as spam that really are spam, and recall is the fraction of spam that gets caught. A false positive, a good email sent to spam, is usually the costlier mistake in a real inbox, so precision is reported first. The Matthews correlation coefficient summarises the whole confusion matrix in one number.

**Reading the results.** The model is slightly more cautious than aggressive. It flags a good email as spam 0.92% of the time and misses 2.06% of spam. Moving the threshold above 0 would lower the false-positive rate further at the cost of recall, and the right trade-off depends on how bad a lost email is for the user. The threshold was not tuned.

**Most informative words.** Ranking words that appear in at least 100 emails by log-odds gives a clear picture of what the model learned:

- Strongest spam signals: `viagra`, `cialis`, `pills`, `pharmacy`, `xanax`, `meds`, `photoshop`, `php`, `serial`, and deliberate misspellings such as `wiil`, `piease` and `shouid`, plus a few tokens that look like address fragments from the corpus.
- Strongest ham signals: `enron`, `kaminski`, `ena`, `hourahead`, `mmbtu`, `westdesk`, `variances`, `scheduling`, `sitara`, `counterparty`, `enrononline`.

The spam side is what one would expect. The ham side is not general knowledge about legitimate mail. It is the vocabulary and the people of one company.


**Caveats that make the numbers optimistic.**

- The split is random, not chronological. A deployed filter sees future mail, and spam changes over time.
- Near-duplicate spam can appear on both sides of the split.
- The ham comes from few mailboxes, and the model partly learns whose mail it is looking at. On a different inbox it would perform worse and would need retraining.
- The corpus covers 1999 to 2005, and modern spam looks different.

## Reproducing the results

Requirements: Python 3 and the packages in `requirements.txt` (`numpy`, `pandas`, `matplotlib`, `nltk`). The notebook was developed on Python 3.13 with numpy 2.5.3, pandas 3.0.5, matplotlib 3.11.2 and nltk 3.10.4, and it also runs on Kaggle's notebook environment.

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
jupyter notebook spam_filter.ipynb
```

Run all cells. The first run needs an internet connection to download the dataset (about 15 MB) and the NLTK stopword list. The dataset is saved as `enron_spam_data.csv` next to the notebook, and `.gitignore` keeps it out of version control.

The pipeline is deterministic. The same confusion matrix came out in two independent environments, a local sandbox and Kaggle. The results will only differ if the stopword list in a different NLTK version differs from the one used here.

## Repository layout

```
.
├── spam_filter.ipynb    full pipeline, with saved outputs
├── requirements.txt     Python dependencies
├── .gitignore           keeps the downloaded dataset out of git
└── README.md
```


## Data credits

The Enron-Spam dataset was collected by V. Metsis, I. Androutsopoulos and G. Paliouras and described in:

> V. Metsis, I. Androutsopoulos and G. Paliouras. *Spam Filtering with Naive Bayes - Which Naive Bayes?* Third Conference on Email and Anti-Spam (CEAS), 2006.

The single-file CSV used here is distributed by [MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data) and is downloaded at runtime from that repository. It is not stored or redistributed here.
