## Machine learning models - Recurrent Neural Networks

Different from the previous sections, this task is sentence classification.
For this task, our previous feature engineering method is not good enough, which make 80% accuracy with tree-based models.
Therefore, we are going to build a bidirectional recurrent neural network to predict the label of a sentence.

Initialize packages and variables
```
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

size = 22560
train_size = int(size * 0.8)
buffer_size = 50000
batch_size = 64

labels = ["company", "tasks", "profile", "benefits"]
tokenizer = tfds.features.text.Tokenizer()
vocabulary_set = set()
labeled_data = []
```

Load data
```
def labeler(text, index):
    return text, tf.cast(index, tf.int64)

def load_data():
    """Load data from raw .txt files, tranform them into
    tf.dataset, split into train and test dataset.

    Returns:
        train_data, test_data (tf.string, tf.int64):
            80% train, 20% test;
            raw text and label {0,1,2,3}
    """
    for i, label in enumerate(labels):
        file_name = "data/" + label + ".txt"
        text_data = tf.data.TextLineDataset(file_name)
        labeled_dataset = text_data.map(lambda ex: labeler(ex, i))
        labeled_data.append(labeled_dataset)

    all_labeled_data = labeled_data[0]
    for labeled_dataset in labeled_data[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

    all_labeled_data = all_labeled_data.shuffle(buffer_size, reshuffle_each_iteration=False)
    train_data = all_labeled_data.take(train_size)
    test_data = all_labeled_data.skip(train_size)

    return train_data, test_data

train_data, test_data = load_data()
```
some examples
```
for data, label in train_data.take(5):
    print(data.numpy(), label.numpy())

>>> b'**Working at Novo Nordisk**  ' 0
>>> b'"With a single monthly membership, companies can help employees find an activity they\'ll love among more than 600 activities across the U.S., Europe, and Latin America."' 0
>>> b'Recruitment plan to make sure that our clients are receiving the best service possible.' 0
>>> b'"Business fluent in English (speaking and writing), Swedish as 2nd preferably language"' 2
>>> b'"Perform sales and marketing analysis, assist with trade shows, literature development, presentations and publicity."' 1
```

Preprocess data - encode, pad
```
def make_vocab(input_data):
    """Get all vocabulary from text tensors

    Args:
        input_data (tuples): text tensor and labels

    Returns:
        vocabulary_set: full vocabulary set.
    """
    for text_tensor, _ in input_data:
        tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(tokens)
    return vocabulary_set

def encode(text_tensor, label):
    """Encode text with tfds.features.text.TokenTextEncoder"""
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_map_fn(text, label):
    """Map function for the tensor data"""
    encoded_text, label = tf.py_function(
        encode,
        inp=[text, label],
        Tout=(tf.int64, tf.int64)
    )
    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label

vocabulary_set = make_vocab(train_data)
encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

train_data_encoded = train_data.map(encode_map_fn)
test_data_encoded = test_data.map(encode_map_fn)

train_data_padded = train_data_encoded.padded_batch(batch_size, padded_shapes=([None],[]))
test_data_padded = test_data_encoded.padded_batch(batch_size, padded_shapes=([None],[]))
```
Some examples after processing
```
for data, label in train_data_padded.take(5):
    print(data.numpy().shape, label.numpy().shape)

>>> (64, 34) (64,)
>>> (64, 62) (64,)
>>> (64, 38) (64,)
>>> (64, 48) (64,)
>>> (64, 41) (64,)
```

Build models
```
def get_uncompiled_model():
    """Build bidirectional RNN model"""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, 64))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.2)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu', name="last_layer"))
    model.add(tf.keras.layers.Dense(4))
    return model

def get_compiled_model():
    """Compile the model"""
    model = get_uncompiled_model()
    model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )
    return model

vocab_size = len(vocabulary_set) + 2
model = get_compiled_model()
```
model summary
```
model.summary()

>>>
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, None, 64)          900160
_________________________________________________________________
bidirectional (Bidirectional (None, 128)               66048
_________________________________________________________________
dense (Dense)                (None, 64)                8256
_________________________________________________________________
last_layer (Dense)           (None, 32)                2080
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 132
=================================================================
Total params: 976,676
Trainable params: 976,676
Non-trainable params: 0
_________________________________________________________________
```

Train model
```
model.fit(
    train_data_padded,
    epochs=5,
    validation_data=test_data_padded,
    validation_steps=30
         )

>>>
Epoch 1/5
282/282 [==============================] - 35s 123ms/step - loss: 0.8148 - accuracy: 0.6799 - val_loss: 0.4716 - val_accuracy: 0.8547
Epoch 2/5
282/282 [==============================] - 33s 117ms/step - loss: 0.3206 - accuracy: 0.9006 - val_loss: 0.3470 - val_accuracy: 0.9010
Epoch 3/5
282/282 [==============================] - 33s 117ms/step - loss: 0.1901 - accuracy: 0.9451 - val_loss: 0.3329 - val_accuracy: 0.9115
Epoch 4/5
282/282 [==============================] - 29s 103ms/step - loss: 0.1312 - accuracy: 0.9649 - val_loss: 0.3807 - val_accuracy: 0.8948
Epoch 5/5
282/282 [==============================] - 27s 97ms/step - loss: 0.1044 - accuracy: 0.9721 - val_loss: 0.3483 - val_accuracy: 0.9026
```

Prediction
```
def encode_pred(text_tensor):
    return np.array(encoder.encode(text_tensor.numpy().decode('utf-8')))

def encode_map_pred(text):
    encoded_text = tf.py_function(
        encode_pred,
        inp=[text],
        Tout=tf.int64
    )
    encoded_text.set_shape([None])
    return encoded_text

def predict(input_list):
    dataset = tf.data.Dataset.from_tensor_slices(input_list)
    data_encoded = dataset.map(encode_map_pred)
    data_padded = data_encoded.padded_batch(batch_size, padded_shapes=([None]))
    y_pred = model.predict_classes(data_padded, batch_size=None)
    return y_pred
```

Customized sentence tokenizer
```
import os
import re
from nltk.data import load

folder = "data/"
tokenizer_file = "english.pickle"
tokenizer_path = os.path.join(folder, tokenizer_file)
tokenizer = load(tokenizer_path)

split_regex = re.compile("(\n)+(?=([-#>*●•·–=\\ ]*[A-Z0-9]|[-#>*●•·–=\\ ]+[A-Z0-9]*))")
newlines_regex = re.compile("(\n)+")
url_regex = re.compile(r"\[[\s\S]*\]\([^)]*")
eg_regex = re.compile(r"e\.g\.|i\.e\.|incl\.")  # e.g. cause wrong tokenized sentence.

def sentence_tokenize(text):
    """Split the full text into sentences."""
    text = eg_regex.sub(" ", text)
    sent_tokens = tokenizer.tokenize(text)
    sent_list = []
    for sent in sent_tokens:
        sub_sent_tokens = split_regex.split(sent)
        sub_sent_tokens = list(
            map(lambda x: url_regex.sub(" ", x), sub_sent_tokens)
        )
        sub_sent_tokens = list(
            filter(lambda x: len(x.split()) > 3, sub_sent_tokens)
        )
        sub_sent_tokens = list(
            map(lambda x: newlines_regex.sub(" ", x), sub_sent_tokens)
        )
        sent_list += sub_sent_tokens
    return sent_list
```

An example of job description from [linkedin](https://www.linkedin.com/jobs/view/1851267491/?alternateChannel=search)
```
sample_text = """Leverage your statistical proficiency and business acumen to champion business innovation in a multi-faceted position. Be at the forefront of data transformation by bringing data science solutions to business functions. Join the Bayer Pharma ‘Data Science & Advanced Analytics’ team within the Division ‘Digital & Commercial Innovation’.

We seek exceptional candidates with strong background in both: Data Science & Business. Successful candidates will demonstrate a high level of statistical knowledge, coupled with strong problem solving and communication skills.

Your Tasks And Responsibilities
Lead projects to provide business executives with analytical solutions to answer global pressing business challenges in various therapeutic areas
Drive competitive advantage of Bayer in making better decisions through the solutions you develop
End-to-end responsibility along the use case cycle: From exploration to model development and insight generation.
Leverage internal and external resources, collaborate with other data scientists, data engineers and analysts to develop innovative solutions
Partner with cross functional stakeholders including marketing, new product commercialization, real world evidence, market access, IT and business intelligence

Who You Are
Advanced degree in a quantitative field, PhD preferred
Four years of experience related to progressive analytics, there of minimum one year in international (pharmaceutical) business or consulting
Strong understanding of machine learning techniques and proficiency with statistical programming (e.g. R, Python)
Experience in transformational leadership, business savvy
Strong communication skills to build collaborative relationships in a diverse, cross-functional environment
Leader and team-player to drive end-to-end implementation of use cases in a structured and time-conscious manner
Business fluent in English, both written and spoken"""
```
```
text_list = sentence_tokenize(sample_text)
predict(text_list)

# display as data frame
pd.DataFrame({
    "label": pred_label,
    "content": text_list
})
```
Result
```
  label	content
0	2	Leverage your statistical proficiency and business acumen to champion business innovation in a multi-faceted position.
1	0	Be at the forefront of data transformation by bringing data science solutions to business functions.
2	0	Join the Bayer Pharma ‘Data Science & Advanced Analytics’ team within the Division ‘Digital & Commercial Innovation’.
3	0	We seek exceptional candidates with strong background in both: Data Science & Business.
4	2	Successful candidates will demonstrate a high level of statistical knowledge, coupled with strong problem solving and communication skills.
5	1	Your Tasks And Responsibilities
6	1	Lead projects to provide business executives with analytical solutions to answer global pressing business challenges in various therapeutic areas
7	2	Drive competitive advantage of Bayer in making better decisions through the solutions you develop
8	1	End-to-end responsibility along the use case cycle: From exploration to model development and insight generation.
9	1	Leverage internal and external resources, collaborate with other data scientists, data engineers and analysts to develop innovative solutions
10	1	Partner with cross functional stakeholders including marketing, new product commercialization, real world evidence, market access, IT and business intelligence
11	2	Advanced degree in a quantitative field, PhD preferred
12	2	Four years of experience related to progressive analytics, there of minimum one year in international (pharmaceutical) business or consulting
13	2	Strong understanding of machine learning techniques and proficiency with statistical programming ( R, Python)
14	2	Experience in transformational leadership, business savvy
15	2	Strong communication skills to build collaborative relationships in a diverse, cross-functional environment
16	1	Leader and team-player to drive end-to-end implementation of use cases in a structured and time-conscious manner
17	2	Business fluent in English, both written and spoken

```

I would say the premier result is quite okay. Further improvements can be:
- find more data, currently we just have small dataset.
- add customized features, check the [feature engineering chapter](https://mlnlp.readthedocs.io/en/latest/Feature-engineering.html)
- use more complex neural network architect, e.g. CNN + RNN, BERT.
- use even more complex architect, neural networks + tree-based classifier.
