import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text

model_path = "../predownload/ted_hrlr_translate_pt_en_converter"

tokenizers = tf.saved_model.load(model_path)

print([item for item in dir(tokenizers.en) if not item.startswith('_')])

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']


x=list(train_examples.batch(3).take(1).as_numpy_iterator())
pt_examples, en_examples=x[0]

for en in en_examples:
  print(en.decode('utf-8'))

encoded = tokenizers.en.tokenize(en_examples)

for row in encoded.to_list():
  print(row)