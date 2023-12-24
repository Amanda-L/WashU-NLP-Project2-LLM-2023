# WashU-NLP-Project2-LSTM-2023

The assignment asked to train an LSTM language model on the Wikitext training corpus, monitor perplexity, and evaluate the model's performance on development and test data.


1. **Find Sentence Length Distribution:**
   - Determine the distribution of sentence lengths in the Wikitext training corpus.
   - Decide on the maximum length (MAXLEN) for training RNN models based on the distribution.

2. **Load Data:**
   - Load the data using the provided code.

3. **Create LSTM Model:**
   - Implement an LSTM model using PyTorch.

```python
# TODO: Set model hyperparameters
embed_dim = None  # Embedding layer size
hidden_dim = None  # LSTM hidden layer size
num_layers = None  # Number of LSTM layers
num_epoch = None  # Maximum training epochs
learning_rate = None  # Learning rate for training
```   - Initialize the model and other necessary variables.

4. **Define Perplexity:**
   - Define perplexity for analysis.

5. **Training the Model:**
   - Set up the training loop.
   - Train the model using backpropagation through time.
   - Detach LSTM states at the end of each epoch.
   - Key steps:
      1. Get the data and run the model.
      2. Update the parameters.
      3. Save the loss and compute perplexity on the validation data.
      4. If the perplexity is higher than the previous epoch, return the model.

6. **Evaluate the Model:**
   - Plot perplexity on the evaluation data against the number of epochs.

## Describe Model and Results



