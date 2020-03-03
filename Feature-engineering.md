## Feature engineering

Load packages and data
```
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD, SparsePCA
from nltk import pos_tag
from nltk.tag import StanfordNERTagger

## make count vectors
df = pd.read_excel("data/sample.xlsx")
samples = df.title.values
```
Prepare vectors
```
vectorizer = CountVectorizer(
  stop_words="english",
  ngram_range=(1,2)
  )

X = vectorizer.fit_transform(samples)
X.shape
```
(269, 1312)

### Dimension reduction
**SVD** <br>
In particular, truncated SVD works on term count/tf-idf matrices as returned by the vectorizers in sklearn.feature_extraction.text. In that context, it is known as latent semantic analysis (LSA). It can work with scipy.sparse matrices efficiently. [Check the mathematical details here](https://nlp.stanford.edu/IR-book/pdf/18lsi.pdf).
```
svd = TruncatedSVD(
  n_components=100,
  n_iter=10,
  random_state=42
  )

X_tran = svd.fit_transform(X)
X_tran.shape
```
(269, 100)

**PCA** <br>
Using PCA for dimensionality reduction is to remove some of the smallest principal components, resulting in a lower-dimensional projection of the data that preserves the maximal data variance. `sklearn.PCA` does not support sparse input, here we use svd combined with sparse_pca instead.

```
spca = SparsePCA(
  n_components=10,
  random_state=0
  )

## first reduced by SVD, then PCA.
X_svd = svd.fit_transform(X)
X_tran = spca.fit_transform(X_svd)

X_tran.shape
```
(269, 10)

Apart from dimension reduction, PCA is a useful tool to deal with correlated features. With PCA, no need to do correlation analysis among features.

### Feature extraction

- **Text-based features**: Number of total/average words, total characters, stopwords, punctuations, UPPER case words, Title case words, unique words, sentences, ...

```
def count_words(text):
    return len(str(text).split())

def count_uniquewords(text):
    return len(set(str(text).split()))

def count_chars(text):
    return len(str(text))

def word_density(text):
    return count_chars(text) / (count_words(text) + 1)

def count_stopwords(text):
    stopwords = [word for word in str(text).split() if word in stop_words]
    return len(stopwords)

def count_pucts(text):
    puncts = re.findall('[' + punctuation + ']', str(text))
    return len(puncts)

def count_upperwords(text):
    upperwords = re.findall(r"\b[A-Z0-9]+\b", str(text))
    return upperwords

def count_firstwords(text):
    """count first word of sentence"""
    firstwords = re.findall(r"\b[A-Z][a-z]+\b", str(text))
    return firstwords
```

- **NLP-based features**: Number of different words: Nouns, Pronouns, Verbs, Adverbs, Adjectives.

```
pos_dic = {
    "NN" : "noun", "NNS" : "noun", "NNP": "noun", "NNPS" : "noun",
    "PRP" : "pron", "PRP$" : "pron", "WP" : "pron", "WP$" : "pron",
    "VB" : "verb", "VBD" : "verb", "VBG" : "verb", "VBN" : "verb", "VBP" : "verb", "VBZ": "verb",
    "JJ" : "adj", "JJR" : "adj", "JJS" : "adj",
    "RB"  : "adv", "RBR" : "adv", "RBS" : "adv", "WRB" : "adj"
}

def count_tag(text):
    pos_counts = {
        "noun": 0, "pron": 0, "verb": 0, "adj": 0, "adv": 0
    }
    for w, p in pos_tag(str(text).split()):
        try:
            tag = pos_dic[p]
            pos_counts[tag] = pos_counts[tag] + 1
        except KeyError:
            pass
    return pos_counts
```

- **NER-based features**: Number of cities/countries/skills/names/company names/...

```
stanford_dir = "/home/ruihaoqiu/stanford-ner-2018-10-16/"
jarfile = stanford_dir + 'stanford-ner.jar'
modelfile = stanford_dir + 'classifiers/english.all.3class.distsim.crf.ser.gz'

st = StanfordNERTagger(model_filename=modelfile, path_to_jar=jarfile)

def count_ner(text):
    ner_counts = dict()
    ners = st.tag(str(text).split())
    print(ners)
    for _, p in ners:
        if p in ner_counts:
            ner_counts[p] = ner_counts[p] + 1
        else:
            ner_counts[p] = 1
    return ner_counts
```
From my testing, I find the Standford NER doesn't work quite well, many wrongly mappings and time consuming. But if it increase the model accuracy, why not use it. Feel free to use any tool e.g. customized NER, to create as many as features.

There are other more sophisticated ways for feature engineering, for example using neural network or CNN, RNN. I will discuss them in other chapters.

**Reference**
- Dimension reduction - [https://thenewstack.io/3-new-techniques-for-data-dimensionality-reduction-in-machine-learning/](https://thenewstack.io/3-new-techniques-for-data-dimensionality-reduction-in-machine-learning/)
- Scikitlearn - [https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca)
- NER (Named Entity Recognition) - [http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford](http://www.nltk.org/api/nltk.tag.html#module-nltk.tag.stanford)
