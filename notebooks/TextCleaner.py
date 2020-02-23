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
