## Embeddings
```
import re
import pandas as pd

from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from TextCleaner import clean_text

## data source - https://www.kaggle.com/bman93/dataset/data#
df = pd.read_csv("data/Top30.csv")
docs = df.Description
len(docs)
```
72292

### Word embedding

**Word2Vec**

clean and tokenize data
```
%%time
tokenss = []
for doc in docs:
    tokenss.append(clean_text(doc))
```
Wall time: 2min 30s, processed ~70k docs.
To generate good work vectors, I would suggest at least 1m jobs. This preprocess will take ~30 mins. If even more data, parallel by PySpark is necessary, see my [pyspark notes](https://pynotes.readthedocs.io/en/latest/pyspark.html).

Train model
```
%%time
model = Word2Vec(
    tokenss,     # list of tokens
    size=500,    # vector length
    window=4,    # maximum distance between the current and predicted word
    min_count=5, # ignores words with frequency lower than 5
    workers=4    # number of threads
)
```
Wall time: 1min 55s

Check results
```
test_list = ["python", "javascript", "powerbi", "excel", "git"]
for word in test_list:
    print(word, ":\n", model.wv.most_similar(word, topn=5))
```
python :
 [('rdbms', 0.8215682506561279), ('perl', 0.8211899399757385), ('tomcat', 0.8167069554328918), ('weblogic', 0.8105891942977905), ('jms', 0.8051434755325317)] </b>  
javascript :
 [('struts', 0.8272807002067566), ('xml', 0.8139920234680176), ('jquery', 0.8123090863227844), ('html', 0.8114718198776245), ('xslt', 0.794143795967102)]</b>  
powerbi :
 [('obiee', 0.6827813982963562), ('tableau', 0.6598259210586548), ('visualization', 0.6336531639099121), ('query', 0.6239848136901855), ('iri', 0.6232795119285583)]</b>  
excel :
 [('ms', 0.666321873664856), ('powerpoint', 0.6542035341262817), ('macros', 0.6245964765548706), ('vlookups', 0.6167137622833252), ('microsoft', 0.6154444217681885)]</b>  
git :
 [('svn', 0.7295184135437012), ('weblogic', 0.6927921772003174), ('tomcat', 0.6880820989608765), ('subversion', 0.6861574053764343), ('jms', 0.6797549724578857)]


### Document embedding

**Doc2Vec**

make tagged documents
```
documents = [TaggedDocument(tokens, [i]) for i, tokens in enumerate(tokenss)]
```
initialize the model, similar as word2vec
```
model = Doc2Vec(
  vector_size=500,
  window=4,
  min_count=5,
  workers=4
  )
```
build vocabulary
```
model.build_vocab(documents)
```
train the model
```
model.train(documents,
            total_examples=model.corpus_count,
            epochs=model.epochs)
```

**Reference**
- Word2Vec explanation - [http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- gensim - [https://radimrehurek.com/gensim/models/word2vec.html](https://radimrehurek.com/gensim/models/word2vec.html)
