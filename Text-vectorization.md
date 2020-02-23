## Text vectorization
How to vectorize the documents.

```
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = pd.read_excel("data/sample.xlsx")
corpus = df.title.values
```

### Bag of words
```
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus)
print(X.shape)
```
(269, 466)
The corpus contains 269 documents, 466 unique words.

### N-gram vectorizer
```
vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(corpus)
print(X.shape)
```
(269, 1312)

### Tfidf vectorizer
tf * idf</b>  
tf - term frequency, word count in a document.</b>  
idf - inverse document frequency, total number of documents / number of documents contain the word.</b>  

The main idea is to lower the weight/importance of the words that appear in many documents.
```
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    sublinear_tf=True
)
X = vectorizer.fit_transform(corpus)
```
It will be the same size as 2-gram vectorization, the values are from 0-1, normalized by L2.

### Customized vectorizer
```
vocab = ["python", "machine learning", "pandas", "pyspark", "sql"]
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    sublinear_tf=True,
    vocabulary = vocab
)
X = vectorizer.fit_transform(corpus)
print(X.shape)
```
(269, 5)</b>  
The documents are only embedded on customized features. An interesting use case, if the features are skills, the values indicate the **importance** of skills in each documents. We can use it to recommend top skills for each documents.

There are other techniques to vectorize the document, I will discuss them in embedding section.
