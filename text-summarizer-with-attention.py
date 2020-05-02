
"""
@author: ritayu
"""

import numpy as np  
import pandas as pd 
import pickle
import re           
from bs4 import BeautifulSoup 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K 
from sklearn.model_selection import train_test_split
# used attention layer implementation from: https://github.com/thushv89/attention_keras/blob/master/layers/attention.py
from attention import AttentionLayer



dat_dir = '/Users/ritayu/Downloads/amazon-fine-food-reviews/'

# loading dataset - around 0.5 million records
reviews_data = pd.read_csv(dat_dir+ 'Reviews.csv')

# dropping all duplicates of 'Text' in reviews_data
reviews_data.drop_duplicates(subset=['Text'],inplace=True)
# dropping all rows with na
reviews_data.dropna(axis=0,inplace=True)

# a glimpse reviews_data in 'Text' and 'Summary' column
reviews_data['Text'].head()
reviews_data['Summary'].head()


# reviews_data preprocessing - 'Text' Cleaning

# found abbrevations dictionary online and using it to clean text
# saved the python dictionary available online in pickle format and loading it
file_open = open(dat_dir+'abv_mapping.pkl','rb')
abv_mapping = pickle.load(file_open)
file_open.close()

# creating a text cleaning function
# importing english stopwords from NLTK corpus
stop_words = set(stopwords.words('english')) 
def text_cleaner(text):
    # lower the string
    updated_string = text.lower()
    # using BeautifulSoup ti identify html tags and removing the tags
    updated_string = BeautifulSoup(updated_string, "lxml").text
    # removing text inside parenthesis
    updated_string = re.sub(r'\([^)]*\)', '', updated_string)
    # removing quotes from text
    updated_string = re.sub('"','', updated_string)
    # treating abbrevations with the mapping
    updated_string = ' '.join([abv_mapping[t] if t in abv_mapping else t for t in updated_string.split(" ")])   
    # removing apostrophe s
    updated_string = re.sub(r"'s\b","",updated_string)
    # removing numbers and other punctuations and special characters
    updated_string = re.sub("[^a-zA-Z]", " ", updated_string) 
    # removing stop words from text
    tokens = [w for w in updated_string.split() if not w in stop_words]
    # 1 or 2 letter words usually dont carry any meaning and removing them from text
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

# generating cleaned text
cleaned_text = []
for text in reviews_data['Text']:
    cleaned_text.append(text_cleaner(text))

# reviews_data preprocessing - 'Summary' Cleaning
# creating similar cleaning function as text cleaner
    
def summary_cleaner(text):
    # lower casing the string
    updated_string = text.lower()
    # removing quotes
    updated_string = re.sub('"','', text)
    # mapping abbrevations
    updated_string = ' '.join([abv_mapping[t] if t in abv_mapping else t for t in updated_string.split(" ")])  
    # removing 's from reviews_data
    updated_string = re.sub(r"'s\b","",updated_string)
    # removing numbers, special characters and other punctuations from reviews_data
    updated_string = re.sub("[^a-zA-Z]", " ", updated_string)
    # considering all words that are more than single alphabet for summary
    tokens=updated_string.split()
    updated_string=''
    for i in tokens:
        if len(i)>1:                                 
            updated_string=updated_string+i+' '  
    return updated_string

# generating cleaned summary
cleaned_summary = []
for summary in reviews_data['Summary']:
    cleaned_summary.append(summary_cleaner(summary))
    
# updating reviews_data
reviews_data['cleaned_text']=cleaned_text
reviews_data['cleaned_summary']=cleaned_summary
reviews_data['cleaned_summary'].replace('', np.nan, inplace=True)
# removing entries where
reviews_data.dropna(axis=0,inplace=True)

# we will add <start> and <end> tokens to the summary for RNN to understand starting and end of summary while training
reviews_data['cleaned_summary'] = reviews_data['cleaned_summary'].apply(lambda x : '_START_ '+ x + ' _END_')

# taking look at cleaned text and summary
for i in range(5):
    print("Review:",reviews_data['cleaned_text'][i])
    print("Summary:",reviews_data['cleaned_summary'][i])
    print("\n")
    
# plotting the length of text and summary to understand the ideal lengths for the RNN model

import matplotlib.pyplot as plt
text_word_count = []
summary_word_count = []

# populate the lists with sentence lengths
for i in reviews_data['cleaned_text']:
      text_word_count.append(len(i.split()))

for i in reviews_data['cleaned_summary']:
      summary_word_count.append(len(i.split()))

length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})
length_df.hist(bins = 30)
plt.show()

reviews_data.to_csv('reviews.csv', index = False)

# majority of sentences land around 80 words and summary around 10 words. We will set these lengths for the model
max_len_text=80 
max_len_summary=10

# using sklearn to split reviews_data into training and testing
x_train,x_val,y_train,y_val=train_test_split(reviews_data['cleaned_text'],reviews_data['cleaned_summary'],test_size=0.1,random_state=0,shuffle=True) 

# Tokenizer from Keras helps build volabulary and converts word sequences into integer sequences

# preparing a tokenizer for 'Text'
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_train))

#convert text sequences into integer sequences
x_train    =   x_tokenizer.texts_to_sequences(x_train) 
x_val   =   x_tokenizer.texts_to_sequences(x_val)

#padding zero upto maximum length if length of text is small
# padding with default trimming to 'pre' to make sure that the 'end' token is always in the reviews_data
x_train    =   pad_sequences(x_train,  maxlen=max_len_text, padding='post') 
x_val   =   pad_sequences(x_val, maxlen=max_len_text, padding='post')

# size of vocabolary + 1 to be feed into embedding layer
x_vocab_size   =  len(x_tokenizer.word_index) +1 


#preparing a tokenizer for 'Summary' on training reviews_data 
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_train))

#convert summary sequences into integer sequences
y_train    =   y_tokenizer.texts_to_sequences(y_train) 
y_val   =   y_tokenizer.texts_to_sequences(y_val) 

#padding zero upto maximum length if length of summary is less than maximum length
y_train    =   pad_sequences(y_train, maxlen=max_len_summary, padding='post')
y_val   =   pad_sequences(y_val, maxlen=max_len_summary, padding='post')

# size of vocabolary + 1 to be feed into embedding layer
y_vocab_size  =   len(y_tokenizer.word_index) +1


# Building the Neural Network
# Using 3 stacked LSTM layers in the encoder part of LSTM
# using 300 dimension for the embedding layer and lstm layer, different models with different latent dimensions can be tried to fine tune the model
embed_dim = 300
lstm_dim = 300

K.clear_session() 

# Encoder 
encoder_inputs = Input(shape=(max_len_text,)) 
# if more time permits, we can use glove or word2vec pre-trained embeddings as well and compare model efficiency
encoder_embed = Embedding(x_vocab_size, embed_dim,trainable=True)(encoder_inputs) 

# implementing Unidirection LSTM, we can try Bidirectional LSTM for improving model accuracy if time permits
#LSTM Layer 1 
encoder_lstm1 = LSTM(lstm_dim,return_sequences=True,return_state=True) 
encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_embed) 

#LSTM Layer 2 
encoder_lstm2 = LSTM(lstm_dim,return_sequences=True,return_state=True) 
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 

#LSTM Layer 3 
encoder_lstm3 = LSTM(lstm_dim, return_state=True, return_sequences=True) 
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2) 

# Set up the decoder. Defining input to take any shape
decoder_inputs = Input(shape=(None,)) 
# embedding the output variable
decoder_embed_layer = Embedding(y_vocab_size, embed_dim,trainable=True) 
decoder_embed = decoder_embed_layer(decoder_inputs) 

#LSTM using encoder_states as initial state
decoder_lstm = LSTM(lstm_dim, return_sequences=True, return_state=True) 
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(decoder_embed,initial_state=[state_h, state_c]) 

#Attention Layer - implementation referenced from internet
# Referenced Papers - https://arxiv.org/abs/1508.04025.pdf, https://arxiv.org/pdf/1409.0473.pdf
attn_layer = AttentionLayer(name='attention_layer') 
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs]) 

# Concat attention output and decoder LSTM output 
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer
decoder_dense = TimeDistributed(Dense(y_vocab_size, activation='softmax')) 
decoder_outputs = decoder_dense(decoder_concat_input) 

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs) 
model.summary()


# compiling model and using rmsprop optimizer and sparse categorical cross-entropy as it converts integer sequence to a one-hot vector
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# adding regularization in terms of early stopping and storing weights for best model
callbcks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
            ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
            ModelCheckpoint(dat_dir+'model-rnn-summary.h5', monitor = 'loss', verbose=1, save_best_only=True, save_weights_only=True)]

# fitting model on reviews_data
results = model.fit([x_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=10,callbacks=callbcks, batch_size=32, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

# loading weights and the model fit results:

model.load_weights(dat_dir+'model-rnn-summary-gpu.h5')

# model results
results = np.load(dat_dir+'rnn-model_results.npy', allow_pickle='TRUE').item()

# vizualizing the model fit results
import matplotlib.pyplot as plt

# extracting loss and accuracy from results
train_loss=results['loss']
val_loss=results['val_loss']
# setting x axis as per number of epochs
xc=range(len(results['loss']))

plt.figure(figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Val Loss')
plt.grid(True)
plt.legend(['Train','Val'])
plt.style.use(['classic'])
plt.show()

# building dictionaries to generate words back from numerical reference
reverse_target_word_index = y_tokenizer.index_word 
reverse_source_word_index = x_tokenizer.index_word 
target_word_index = y_tokenizer.word_index

# we need encoder and decoder model for  prediction, therefore building the encoder and decoder model separately
# encoder inference for prediction
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# decoder inference for prediction
# Below objects will hold the states of the previous time step
decoder_state_input_h = Input(shape=(lstm_dim,))
decoder_state_input_c = Input(shape=(lstm_dim,))
decoder_hidden_state_input = Input(shape=(max_len_text,lstm_dim))

# Get the embeddings of the decoder sequence
decoder_embed2= decoder_embed_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_embed2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob distribution over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
[decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
[decoder_outputs2] + [state_h2, state_c2])

# defining a function that uses the above

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    encode_out, encode_h, encode_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, decode_h, decode_c = decoder_model.predict([target_seq] + [encode_out, encode_h, encode_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!='end'):
            decoded_sentence += ' '+sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary-1)):
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        encode_h, encode_c = decode_h, decode_c

    return decoded_sentence

# defining function for convert integer sequence to summary
def seq2summary(input_seq):
    updated_string=''
    for i in input_seq:
      if((i!=0 and i!=target_word_index['start']) and i!=target_word_index['end']):
        updated_string=updated_string+reverse_target_word_index[i]+' '
    return updated_string

# defining function for convert integer sequence to text
def seq2text(input_seq):
    updated_string=''
    for i in input_seq:
      if(i!=0):
        updated_string=updated_string+reverse_source_word_index[i]+' '
    return updated_string

# Model in action:
for i in range(len(x_val)):
  print("Review:",seq2text(x_val[i]))
  print("Original summary:",seq2summary(y_val[i]))
  print("Predicted summary:",decode_sequence(x_val[i].reshape(1,max_len_text)))
  print("\n")


