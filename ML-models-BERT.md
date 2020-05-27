## Machine learning models - BERT

In this chapter, let's try BERT.

### Background
BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art technique for NLP pre-training developed by Google in 2018. It is the first deeply bidirectional, unsupervised language representation, pre-trained using only a plain text corpus. It achieved state-of-the-art performance on many NLP tasks[1]. Google has already applied it for query searching, claimed that the search improvement by BERT as "one of the biggest leaps forward in the history of Search"[2].

In this chapter, we will continue on text classification task as the previous chapters, to see how BERT can improve the performance.

### Codes
Full code can be find in the `notebooks/BERT-classifier.ipynb`. The notebook should run on TPU on Google Colab.

- Load pre-train model
```
!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip uncased_L-12_H-768_A-12.zip

bert_folder = "uncased_L-12_H-768_A-12/"
bert_config_file = os.path.join(bert_folder, "bert_config.json")
bert_ckpt_file = os.path.join(bert_folder, "bert_model.ckpt")
bert_vocab_file = os.path.join(bert_folder, "vocab.txt")
```

- Preprocessing
```
tokenizer = FullTokenizer(vocab_file=bert_vocab_file)
max_len = 64

def _convert_single(input_text):
  tokens = tokenizer.tokenize(input_text)
  tokens = ["[CLS]"] + tokens + ["[SEP]"]
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  return token_ids

def _convert_multiple(input_list):
  token_ids_list = []
  max_len = 1
  for sent in tqdm(input_list):
    token_ids = _convert_single(sent)
    token_ids_list.append(token_ids)
  return token_ids_list

def _pad(token_ids_list):
  x_padded = []
  for input_ids in token_ids_list:
    input_ids = input_ids[:min(len(input_ids), max_len - 2)]
    input_ids = input_ids + [0] * (max_len - len(input_ids))
    x_padded.append(np.array(input_ids))
  return np.array(x_padded)

def convert(input_list):
  token_ids_list = _convert_multiple(input_list)
  out_array = _pad(token_ids_list)
  return out_array

X_train = convert(train.sentence)
X_test = convert(test.sentence)

y_train = train.label.map(label_dict).values
y_test = test.label.map(label_dict).values
```

- Create model
```
def create_model():

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")

  input_ids = tf.keras.layers.Input(
    shape=(max_len, ),
    dtype='int32',
    name="input_ids"
  )
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = tf.keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = tf.keras.layers.Dropout(0.5)(cls_out)
  logits = tf.keras.layers.Dense(units=768, activation="tanh")(cls_out)
  logits = tf.keras.layers.Dropout(0.5)(logits)
  logits = tf.keras.layers.Dense(
    units=len(classes),
    activation="softmax"
  )(logits)

  model = tf.keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_len))

  load_stock_weights(bert, bert_ckpt_file)

  return model

model = create_model()

model.compile(
  optimizer=tf.keras.optimizers.Adam(1e-5),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
)
```
Train model
```
history = model.fit(
  x=X_train,
  y=y_train,
  validation_split=0.2,
  batch_size=32,
  shuffle=True,
  epochs=10,
  callbacks=[tensorboard_callback]
)
```

References:
1. BERT paper - https://arxiv.org/pdf/1810.04805.pdf
2. Google blog apply BERT for searching - https://www.blog.google/products/search/search-language-understanding-bert/
