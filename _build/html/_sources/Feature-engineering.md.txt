## Feature engineering

```
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD, SparsePCA

## make count vectors
df = pd.read_excel("data/sample.xlsx")
samples = df.title.values
vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(samples)
X.shape
```
(269, 1312)

### Dimension reduction
**SVD**
In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA). It can work with scipy.sparse matrices efficiently. [Check the mathematical details here](https://nlp.stanford.edu/IR-book/pdf/18lsi.pdf).
```
svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
X_tran = svd.fit_transform(X)
print(X_tran.shape)
```
(269, 100)

**PCA**
Using PCA for dimensionality reduction is to remove some of the smallest principal components, resulting in a lower-dimensional projection of the data that preserves the maximal data variance. `sklearn.PCA` does not support sparse input, here we use svd combined with sparse_pca instead.

```
spca = SparsePCA(n_components=10, random_state=0)
X_svd = svd.fit_transform(X)
X_tran = spca.fit_transform(X_svd)
print(X_tran.shape)
```
(269, 10)


### Feature extraction

- **text-based features**: Number of total/average words, total characters, stopwords, punctuations, UPPER case words, Title case words, unique words, sentences, ...
- **NLP-based features**: Number of different words: Nouns, Pronouns, Verbs, Adverbs, Adjectives.
- **NER-based features**: Number of cities/countries/skills/names/company names/...

**Reference**
- Dimension reduction - [https://thenewstack.io/3-new-techniques-for-data-dimensionality-reduction-in-machine-learning/](https://thenewstack.io/3-new-techniques-for-data-dimensionality-reduction-in-machine-learning/)
- Scikitlearn - [https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca)
- NER (Named Entity Recognition) - [http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford](http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford)
