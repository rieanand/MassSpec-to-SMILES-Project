import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Conv1D, MaxPooling1D, BatchNormalization,
    Dropout, Bidirectional, Add, Concatenate, TimeDistributed
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sequence_to_smiles_model(input_shape, vocabulary_size, embedding_dimension=256, latent_dimension=512):
    """Builds a sequence-to-sequence model for SMILES generation from mass spectrometry data"""

    # encoder
    encoder_input_layer = Input(shape=input_shape)
    normalized_input_layer = BatchNormalization()(encoder_input_layer)

    # convolutional layers
    """
    Kernel Size: 3 
    Filters: embedding dimension (256)
    Activation Function: ReLU  
    Padding: 'same' to ensure output sequence length remains consistent
    Batch Normalization: normalization for stabilization and convergence
    Residual Connections: "Add" is used to combine outputs of the current and previous convolution layers
    MaxPooling (pool size=2): pooling is used to reduce the size of the feature map (reducing computational complexity)
    Dropout (0.1): used to regularize the model and prevent overfitting 
    """
    first_convolution_layer = Conv1D(embedding_dimension, 3, activation="relu", padding="same")(normalized_input_layer)
    first_convolution_output = Conv1D(embedding_dimension, 3, activation="relu", padding="same")(BatchNormalization()(first_convolution_layer))
    first_pooling_layer = MaxPooling1D(2)(Dropout(0.1)(Add()([BatchNormalization()(first_convolution_layer), first_convolution_output])))

    second_convolution_layer = Conv1D(embedding_dimension, 3, activation="relu", padding="same")(first_pooling_layer)
    second_convolution_output = Conv1D(embedding_dimension, 3, activation="relu", padding="same")(BatchNormalization()(second_convolution_layer))
    second_pooling_layer = MaxPooling1D(2)(Dropout(0.1)(Add()([BatchNormalization()(second_convolution_layer), second_convolution_output])))

    # bidirectional lstm
    """
    Latent Dimension: the hidden state is set to half the latent dimension to ensure that the forward and backward states concatenated match the latent dimension
    return_sequences=True: outputs the sequence for the decoder input
    return_state=True: outputs the final hidden and cell states. This is passed as the initial state for the decoder
    
    Outputs:
        forward_hidden_state and backward_hidden_state: hidden states from the forward and backward LSTMs
        forward_cell_state and backward_cell_state: cell states from the forward and backward LSTMs
        
    Concatenation: combines the forward and backward hidden and cell states to form the initial states for the decoder
    """
    encoder_bidirectional_lstm = Bidirectional(LSTM(latent_dimension // 2, return_sequences=True, return_state=True))
    _, forward_hidden_state, forward_cell_state, backward_hidden_state, backward_cell_state = encoder_bidirectional_lstm(second_pooling_layer)
    encoder_combined_states = [Concatenate()([forward_hidden_state, backward_hidden_state]),
                                Concatenate()([forward_cell_state, backward_cell_state])]
    
    # decoder
    """
    Input Layer: takes a sequence of tokens representing the decoder's input (previously generated SMILES tokens)
    Embedding Layer: maps discrete tokens into dense vectors of fixed dimensionality (e.g., 256), providing a continuous representation of the input sequence.
    LSTM Layer: processes the embedded input sequence, using the combined states from the Bidirectional LSTM as its initial hidden and cell states.
    TimeDistributed Dense Layer: applies a dense layer to each step of the lstm output and outputs a probability distribution over the vocabulary size for each token poisiton. 
    Dropout (0.3): used to regularize the model and prevent overfitting 
    Softmax Activation: for multi-class prediction
    Final Output: predicts the subsequent tokens in the SMILES sequence for each position 
    """

    decoder_input_layer = Input(shape=(None,))
    decoder_embedding_layer = Embedding(vocabulary_size, embedding_dimension)(decoder_input_layer)
    decoder_lstm_layer = LSTM(latent_dimension, return_sequences=True)(decoder_embedding_layer, initial_state=encoder_combined_states)
    decoder_output_layer = TimeDistributed(Dense(vocabulary_size, activation='softmax'))(Dropout(0.3)(decoder_lstm_layer))

    return Model([encoder_input_layer, decoder_input_layer], decoder_output_layer)


def preprocess_ms(spectrum_string):
    """Preprocesses a mass spectrum string into a list of (m/z, intensity) tuples"""
    try:
        df = pd.DataFrame([peak.split(':') for peak in spectrum_string.split()], columns=['m/z', 'intensity'])
        df = df.astype(float)  # convert columns to float
        df['intensity'] /= df['intensity'].max()  # normalize intensity values
        return list(df.itertuples(index=False, name=None))
    except (ValueError, IndexError):
        return []  # return an empty list if parsing fails


def build_vocabulary(dataframe):
    """Builds a vocabulary from SMILES strings"""
    tokenized_smiles_strings = dataframe['SMILES'].apply(list)
    vocabulary = sorted(set(token for tokens in tokenized_smiles_strings for token in tokens))
    token_to_index_map = {character: index + 1 for index, character in enumerate(vocabulary)}
    index_to_token_map = {index + 1: character for index, character in enumerate(vocabulary)}
    return token_to_index_map, index_to_token_map


def main():
    # data loading
    data_file = 'cleaned_gc_spectra.csv'
    df = pd.read_csv(data_file).dropna(subset=['Spectrum', 'SMILES'])
    df['SMILES'] = df['SMILES'].astype(str)

    # process spectra
    df['Processed_Spectrum'] = df['Spectrum'].apply(preprocess_ms)
    df = df[df['Processed_Spectrum'].apply(len) > 0]

    # build vocabulary from SMILES
    token_to_index, index_to_token = build_vocabulary(df)

    # tokenize SMILES
    tokenized_smiles = df['SMILES'].apply(lambda x: [token_to_index.get(tok, 0) for tok in list(x)])
    padded_smiles = pad_sequences(tokenized_smiles, padding='post')
    vocab_size = len(token_to_index) + 1

    # create spectrum array
    max_length = max(len(spectrum) for spectrum in df['Processed_Spectrum'])
    spectra_array = np.array([
        [intensity for _, intensity in spectrum] + [0] * (max_length - len(spectrum))
        for spectrum in df['Processed_Spectrum']
    ])
    spectra_array = StandardScaler().fit_transform(spectra_array)

    # split datasets
    X_train, X_test, y_train, y_test = train_test_split(spectra_array, padded_smiles, test_size=0.2, random_state=42)
    X_train_expanded = np.expand_dims(X_train, axis=-1)
    X_test_expanded = np.expand_dims(X_test, axis=-1)

    # prepare decoder inputs and outputs
    decoder_input_train = y_train[:, :-1]
    y_train_one_hot = to_categorical(y_train[:, 1:], num_classes=vocab_size)

    # build and compile model
    input_shape = (X_train_expanded.shape[1], X_train_expanded.shape[2])
    model = sequence_to_smiles_model(input_shape, vocab_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # train model
    model.fit([X_train_expanded, decoder_input_train], y_train_one_hot, batch_size=32, epochs=50,
              validation_split=0.1,
              callbacks=[
                  EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                  ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
              ])

    # generate predictions
    y_test_preds = model.predict([X_test_expanded, y_test[:, :-1]])

    # decode predictions
    test_decoded = [
        ''.join(index_to_token.get(np.argmax(token), '') for token in seq)
        for seq in y_test_preds
    ]
    actual_decoded = [
        ''.join(index_to_token.get(tok, '') for tok in seq if tok > 0)
        for seq in y_test
    ]

    # save predictions
    print("Saving predictions...")
    pd.DataFrame({'actual': actual_decoded, 'predicted': test_decoded}).to_csv('predictions.csv', index=False)


if __name__ == "__main__":
    main()
