## Data cleaning

```
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from Trie import Trie

## input an text example - a job post
filename = 'data/text_example.txt'
with open(filename, 'rt') as handler:
    text = handler.read()

## stop words and punctuations
stop_words = stopwords.words('english')
punc = string.punctuation
```

### Clean text with regex
**Clean special patterns**
```
def clean_special_patterns(text):
    """Remove special patterns - email, url, date etc."""
    email_regex = re.compile(r"[\w.-]+@[\w.-]+")
    url_regex = re.compile(r"(http|www)[^\s]+")
    date_regex = re.compile(r"[\d]{2,4}[ -/:]*[\d]{2,4}([ -/:]*[\d]{2,4})?") # a way to match date
    ## remove
    text = url_regex.sub("", text)
    text = email_regex.sub("", text)
    text = date_regex.sub("", text)
    return text

s = """Applications:
www.aa.frdfaunefehofer.de/defe/referfefenzenefe/afeda-cenfeter.html
http://www.ifefis.fe.com
email: fowjfoj@fwjofj.djfow
Kennziffer: IIS-2020-12-23
Bewerbungsfrist:
"""
print(clean_special_patterns(s).strip)
```
'Applications: \n\n\nemail: \nKennziffer: IIS-\nBewerbungsfrist:\n'

**Remove stopwords and punctions**
```
def make_regex(input_list):
    """Build regex from trie structure.
    """
    t = Trie()
    for w in input_list:
        t.add(w)
    regex = re.compile(r"\b" + t.pattern() + r"\b", re.IGNORECASE)
    return regex

def clean_stopwords(text):
    stop_regex = make_regex(stop_words)
    text = stop_regex.sub("", text)
    return text

def clean_punct(text):
    punc_regex = re.compile('[%s]'%re.escape(string.punctuation))
    text = punc_regex.sub("", text)
    return text

clean_stopword(text)
clean_punct(text)
```
About Trie data structure, [check my other post](https://algonotes.readthedocs.io/en/latest/Trie.html).
The script Trie.py, which you can find [here](https://gist.github.com/EricDuminil/8faabc2f3de82b24e5a371b6dc0fd1e0).


### Clean text with NLTK
**Tokenize**
```
tokens = word_tokenize(text)
```
**Remove punctuations**
```
words = [word.lower() for word in tokens if word.isalpha()]
```
**Remove stop words**
```
stop_words = stopwords.words('english')
words = [word for word in words if not word in stop_words]
```

**Stemming**
```
porter = PorterStemmer()
stemmed_words = [porter.stem(word) for word in words]
```

### Text cleaning pipeline
```
def clean_text(text):
    """clean text by
    clean_special_patterns: email, date, url, etc.
    remove punctions, stop words
    stem words

    output
    --------
    list: stemmed words
    """
    s = clean_special_patterns(text)
    tokens = word_tokenize(text)
    words = [word.lower() for word in tokens if word.isalpha()]
    words = [word for word in words if not word in stop_words]
    stemmed_words = [porter.stem(word) for word in words]
    return stemmed_words
```

### Text cleaning module
Build a cleaning module based on the above contents.
```
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from Trie import make_regex

## stop words and punctuations
stop_words = stopwords.words('english')

## regex
email_regex = re.compile(r"[\w.-]+@[\w.-]+")
url_regex = re.compile(r"(http|www)[^\s]+")
date_regex = re.compile(r"[\d]{2,4}[ -/:]*[\d]{2,4}([ -/:]*[\d]{2,4})?") # a way to match date
keep_word_regex = re.compile(r"[^A-Za-z ]+")
stop_regex = make_regex(stop_words)

def clean_special_patterns(text):
    """Remove special patterns - email, url, date etc."""
    ## remove
    text = url_regex.sub("", text)
    text = email_regex.sub("", text)
    text = date_regex.sub("", text)
    return text

def clean_stopwords(text):
    text = stop_regex.sub("", text)
    return text

def clean_keep_words(text):
    return keep_word_regex.sub(" ", text)

def clean_text(text):
    text = clean_special_patterns(text)
    text = clean_stopwords(text)
    text = clean_keep_words(text)
    tokens = [word.lower() for word in word_tokenize(text)]
    return tokens
```

To be notice that, there is no universal text cleaning method. For some classification tasks, special characters might be good features, they should not be removed. For word2vec task, it is better not to stem the words and some stop words maybe important. For text generation, stop words might be also useful.
