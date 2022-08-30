# Text GENERATOR
 DEVELOPED BY RISHU
pip install keras_nlp
keras_nlp.__version__
# 0.3.0
#Implementing a GPT-Style Model with Keras
import tensorflow as tf
from tensorflow import keras
import keras_nlp
import numpy as np
# Loading Data
crime_and_punishment_url = 'https://www.gutenberg.org/files/2554/2554-0.txt'
brothers_of_karamazov_url = 'https://www.gutenberg.org/files/28054/28054-0.txt'
the_idiot_url = 'https://www.gutenberg.org/files/2638/2638-0.txt'
the_possessed_url = 'https://www.gutenberg.org/files/8117/8117-0.txt'

paths = [crime_and_punishment_url, brothers_of_karamazov_url, the_idiot_url, the_possessed_url]
names = ['Crime and Punishment', 'Brothers of Karamazov', 'The Idiot', 'The Possessed']
texts = ''
for index, path in enumerate(paths):
    filepath = keras.utils.get_file(f'{names[index]}.txt', origin=path)
    text = ''
    with open(filepath, encoding='utf-8') as f:
        text = f.read()
        # First 50 lines are the Gutenberg intro and preface
        # Skipping first 10k characters for each book should be approximately
        # removing the intros and prefaces.
        texts += text[10000:]
        # 500 characters
texts[25000:25500]
'nd that was why\nI addressed you at once. For in unfolding to you the story of my life, I\ndo not wish to make myself a laughing-stock before these idle listeners,\nwho indeed know all about it already, but I am looking for a man\nof feeling and education. Know then that my wife was educated in a\nhigh-class school for the daughters of noblemen, and on leaving she\ndanced the shawl dance before the governor and other personages for\nwhich she was presented with a gold medal and a certificate of merit.\n'
text_list = texts.split('.')
len(text_list) # 69181
# Filter out empty strings ('') that are to be found commonly due to the book's format
text_list = list(filter(None, text_list))

import random
random.shuffle(text_list)
length = len(text_list)
text_train = text_list[:int(0.7*length)]
text_test = text_list[int(0.7*length):int(0.85*length)]
text_valid = text_list[int(0.85*length):]
length = len(text_list)
text_train = text_list[:int(0.7*length)]
text_test = text_list[int(0.7*length):int(0.85*length)]
text_valid = text_list[int(0.85*length):]
# Text Vectorization
...
sequence = ['I', 'am', 'Wall-E']
sequence = tokenize(sequence)
print(sequence) # [4, 26, 472]
...
sequence = pad_sequence(sequence)
print(sequence) # [4, 26, 472, 0, 0]
from tensorflow.keras.layers import TextVectorization

def custom_standardization(input_string):
    sentence = tf.strings.lower(input_string)
    sentence = tf.strings.regex_replace(sentence, "\n", " ")
    return sentence

maxlen = 50
# You can also set calculate the longest sentence in the data - 25 in this case
# maxlen = len(max(text_list).split(' ')) 

vectorize_layer = TextVectorization(
    standardize = custom_standardization,
    output_mode="int",
    output_sequence_length=maxlen + 1,
)

vectorize_layer.adapt(text_list)
vocab = vectorize_layer.get_vocabulary()
vocab_size = len(vocab)
vocab_size # 49703
index_lookup = dict(zip(range(len(vocab)), vocab))    
index_lookup[5] # of
vectorize_layer(['hello world!'])
<tf.Tensor: shape=(1, 51), dtype=int64, numpy=
array([[   1, 7509,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0]], dtype=int64)>
           # Dataset Creation
           batch_size = 64

train_dataset = tf.data.Dataset.from_tensor_slices(text_train)
train_dataset = train_dataset.shuffle(buffer_size=256)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices(text_test)
test_dataset = test_dataset.shuffle(buffer_size=256)
test_dataset = test_dataset.batch(batch_size)

valid_dataset = tf.data.Dataset.from_tensor_slices(text_valid)
valid_dataset = valid_dataset.shuffle(buffer_size=256)
valid_dataset = valid_dataset.batch(batch_size)
def preprocess_text(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


train_dataset = train_dataset.map(preprocess_text)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = test_dataset.map(preprocess_text)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

valid_dataset = valid_dataset.map(preprocess_text)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
for entry in train_dataset.take(1):
    print(entry)
    (<tf.Tensor: shape=(64, 50), dtype=int64, numpy=
array([[17018,   851,     2, ...,     0,     0,     0],
       [  330,    74,     4, ...,     0,     0,     0],
       [   68,   752, 30273, ...,     0,     0,     0],
       ...,
       [    7,    73,  2004, ...,     0,     0,     0],
       [   44,    42,    67, ...,     0,     0,     0],
       [  195,   252,   102, ...,     0,     0,     0]], dtype=int64)>, <tf.Tensor: shape=(64, 50), dtype=int64, numpy=
array([[  851,     2,  8289, ...,     0,     0,     0],
       [   74,     4,    34, ...,     0,     0,     0],
       [  752, 30273,  7514, ...,     0,     0,     0],
       ...,
       [   73,  2004,    31, ...,     0,     0,     0],
       [   42,    67,    76, ...,     0,     0,     0],
       [  252,   102,  8596, ...,     0,     0,     0]], dtype=int64)>)
       # Model Definition
       embed_dim = 128
num_heads = 4

def create_model():
    inputs = keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(vocab_size, maxlen, embed_dim)(inputs)
    decoder = keras_nlp.layers.TransformerDecoder(intermediate_dim=embed_dim, 
                                                            num_heads=num_heads, 
                                                            dropout=0.5)(embedding_layer)
    
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(decoder)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer="adam", 
        loss='sparse_categorical_crossentropy',
        metrics=[keras_nlp.metrics.Perplexity(), 'accuracy']
    )
    return model

model = create_model()
model.summary()
Model: "model_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_6 (InputLayer)        [(None, 30)]              0         
                                                                 
 token_and_position_embeddin  (None, 30, 128)          6365824   
 g_5 (TokenAndPositionEmbedd                                     
 ing)                                                            
                                                                 
 transformer_decoder_5 (Tran  (None, 30, 128)          132480    
 sformerDecoder)                                                 
                                                                 
 dense_5 (Dense)             (None, 30, 49703)         6411687   
                                                                 
=================================================================
Total params: 13,234,315
Trainable params: 13,234,315
Non-trainable params: 0
def create_model():
    inputs = keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    x = keras_nlp.layers.TokenAndPositionEmbedding(vocab_size, maxlen, embed_dim)(inputs)
    for i in range(4):
        x = keras_nlp.layers.TransformerDecoder(intermediate_dim=embed_dim*2, num_heads=num_heads,                                                             dropout=0.5)(x)
    do = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(vocab_size, activation='softmax')(do)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    Total params: 13,631,755
Trainable params: 13,631,755
Non-trainable params: 0
class TextSampler(keras.callbacks.Callback):
    def __init__(self, start_prompt, max_tokens):
        self.start_prompt = start_prompt
        self.max_tokens = max_tokens
        
    # Helper method to choose a word from the top K probable words with respect to their probabilities
    # in a sequence
    def sample_token(self, logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def on_epoch_end(self, epoch, logs=None):
        decoded_sample = self.start_prompt
        
        for i in range(self.max_tokens-1):
            tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
            predictions = self.model.predict([tokenized_prompt], verbose=0)
            # To find the index of the next word in the prediction array.
            # The tokenized prompt is already shorter than the original decoded sample
            # by one, len(decoded_sample.split()) is two words ahead - so we remove 1 to get
            # the next word in the sequence
            sample_index = len(decoded_sample.strip().split())-1
            
            sampled_token = self.sample_token(predictions[0][sample_index])
            sampled_token = index_lookup[sampled_token]
            decoded_sample += " " + sampled_token
            
        print(f"\nSample text:\n{decoded_sample}...\n")

# First 5 words of a random sentence to be used as a seed
random_sentence = ' '.join(random.choice(text_valid).replace('\n', ' ').split(' ')[:4])
sampler = TextSampler(random_sentence, 30)
reducelr = keras.callbacks.ReduceLROnPlateau(patience=10, monitor='val_loss')
model = create_model()
history = model.fit(train_dataset, 
                    validation_data=valid_dataset,
                    epochs=10, 
                    callbacks=[sampler, reducelr])
                    # Epoch training
Epoch 1/10
658/658 [==============================] - ETA: 0s - loss: 2.7480 - perplexity: 15.6119 - accuracy: 0.6711
# on_epoch_end() sample generation
Sample text:
”  “What do you had not been i had been the same man was not be the same eyes to been a whole man and he did a whole man to the own...
# Validation
658/658 [==============================] - 158s 236ms/step - loss: 2.7480 - perplexity: 15.6119 - accuracy: 0.6711 - val_loss: 2.2130 - val_perplexity: 9.1434 - val_accuracy: 0.6864 - lr: 0.0010
...
Sample text:
”  “What do you know it is it all this very much as i should not have a great impression  in the room to be  able of it in my heart...

658/658 [==============================] - 149s 227ms/step - loss: 1.7753 - perplexity: 5.9019 - accuracy: 0.7183 - val_loss: 2.0039 - val_perplexity: 7.4178 - val_accuracy: 0.7057 - lr: 0.0010
# Model Inference
def sample_token(logits):
        logits, indices = tf.math.top_k(logits, k=5, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

def generate_text(prompt, response_length=20):
    decoded_sample = prompt
    for i in range(response_length-1):
        tokenized_prompt = vectorize_layer([decoded_sample])[:, :-1]
        predictions = model.predict([tokenized_prompt], verbose=0)
        sample_index = len(decoded_sample.strip().split())-1

        sampled_token = sample_token(predictions[0][sample_index])
        sampled_token = index_lookup[sampled_token]
        decoded_sample += " " + sampled_token
    return decoded_sample
    generate_text('the truth ultimately is')
# 'the truth ultimately is really a joy in history, this state of life through which is almost invisible, superfluous  teleological'

generate_text('the truth ultimately is')
# 'the truth ultimately is not to make it a little   thing to go into your own  life for some'
