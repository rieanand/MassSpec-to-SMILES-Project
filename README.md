### Generating Novel Compounds from Mass Spectrometry Data

##### Objective

This project aims to develop a machine learning model capable of generating a range of possible compounds directly from Mass Spectrometry data.

##### Background

GC-MS is typically used to identify the components of complex mixtures by comparing their mass spectra to those in existing libraries. However, this method is limited by the availability of known spectra. We can expand this capability by developing a model to predict molecular structures from their MS profiles, potentially identifying novel compounds or those not yet cataloged.

##### Data

Our project will use the data available in the Massbank of North America (MoNA), which includes pairs of SMILES strings and their associated mass spectrometry data (M/Z: Intensity) pairs.

https://mona.fiehnlab.ucdavis.edu/

##### First Phase: EDA and Data Cleaning

It is difficult to train our model on MS to SMILES data if there is ambiguous stereochemistry. To avoid this issue, we standardized our SMILES data by using RDKit packages to sanitize, strip isomeric ambiguities, and to canonicalize the SMILES strings. This process is documented in the canonize.ipynb notebook. 
SMILES above 75 in character length and those with characters not present in our target dataset were filtered out. Spectrum length and spectrum m/z frequency distributions are visualized, and these will be used to make decisions about spectrum features for the spectrum to smiles model. 

##### Second Phase: SMILES randomized to canonical strings autoencoder model

Refer to SMILES_AutoEncoder.ipynb file for progress.

- Implement SMILES bidirectional GRU autoencoder. This model will learn to compress SMILES randomized strings and decode them into canonized SMILES strings. This model is used to pre-train the decoder component for the MS to SMILES model since there is a limited amount of available spectra data for training the spectra to SMILES model end-to-end. By using the SMILES molecule datasets, the decoder learns many chemical structures and SMILES string encodings.The spectra encoder will be trained on a smaller labeled dataset of molecules and mass spectra. This approach allows the decoder to learn latent space representation of molecules which will be useful to reconstruct SMILES strings for spectra to molecular structure translation. This ensures that the model can accurately reconstruct molecular structure information, especially with novel molecules not present in databases and our training dataset.

- Bidirectional GRUs are used because of the sequential nature of the SMILES string where we would want to capture contextual information from both directions (forward and reverse) when encoding and decoding strings. Each character has significance in relation to preceding and succeeding elements. GRUs, which is a type of RNN, processes sequential data by capturing dependencies between elements in sequences.

- We have implemented the bidirectional GRU autoencoder using PyTorch with a small sample of five randomly selected SMILES randomized strings. We used cross entropy loss between the model output and target canonized SMILES strings for loss optimization in training. Hyperparameters included sequence length of 77 (length of SMILES string), input size of 64 (unique characters in SMILES string), embedding dimensions of 10 (number of input dimensions after embedding and before encoder), hidden dimensions of 10 (dimensions of hidden layer), number of layers of 3 (layers in encoder and decoder), batch size of 5, and learning rate of 0.0001 for optimization.

- Prior to using the model, the SMILES strings were tokenized and embedded into lists of numbers and then converted to tensors.

- The loss after training epochs was 4.2359 for five SMILES sequences.

##### Third Phase: Spectra Encoder

Following establishing the SMILES autoencoder, we will focus on training a spectra encoder. This encoder aims to generate embeddings from mass spectrometry data that closely correspond to the embeddings created by the SMILES encoder for the same chemical compound. By minimizing the difference in embedding between spectra and SMILES, we enable the SMILES decoder to interpret these spectra-derived embeddings to generate valid SMILES strings, effectively bridging the gap between MS data and molecular identity.

##### Transformer

- Implemented preliminary draft of transformer for spectra embedding in the Transformer.ipynb file. Our next step is to test this with the GRU decoder.
  
- General architecture is as follows:
Self-attention with a Multi-head Attention mechanism, a feed forward layer, multiple normalizations layers, a dropout layer for regularization, a positional encoding layer, an output layer

##### Validation

- After generating compounds, metrics such as molecular weight difference, Tanimoto similarity, and element comparisons will validate the model’s accuracy. These metrics help ensure the generated compounds are structurally plausible and chemically relevant to the input mass spectrometry data.
