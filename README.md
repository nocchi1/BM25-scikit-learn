# Okapi-BM25 Scikit-learn Implementation

This repository provides an implementation of the Okapi BM25 algorithm following the scikit-learn API standards. \
This implementation can be used in the same way as TfidfVectorizer or CountVectorizer.

## Environment
- Python version: 3.10.11
- scikit-learn version: 1.5.2

## Usage
```python

vectorizer = BM25Vectorizer(
    ngram_range=(1, 2),
    k=1.2,
    b=0.75,
    sqrt=False
)

X_train = vectorizer.fit_transform(train_texts)
X_valid = vectorizer.transform(valid_texts)
```

## Parameters
In addition to the parameters available in TfidfVectorizer, the following parameters can be specified
- `k1` (float, default=1.2) : This parameter adjusts the saturation of term frequency.
- `b` (float, default=0.75) : This parameter controls the length normalization of the document.
- `sqrt_tf` (bool, default=True) : This parameter specifies whether to apply square root scaling to the term frequency.
