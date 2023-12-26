# WashU-NLP-Project2-LSTM

The assignment asked to train an LSTM language model on the Wikitext training corpus, monitor perplexity, and evaluate the model's performance on development and test data.


1. **Find Sentence Length Distribution:**
   - Determine the distribution of sentence lengths in the Wikitext training corpus.
   - Decide on the maximum length (MAXLEN) for training RNN models based on the distribution.
```python
MAXLEN =  268 # 628 was taking forever to run and the performance was not good, so I changed it to 268 for lower perplexity.
batch_size = 32 # For training
``` 


2. **Create LSTM Model:**
   - Implement an LSTM model using PyTorch.
   - Initialize the model and other necessary variables.

```
embed_dim = 128 # Embedding layer size
hidden_dim = 64 # (LSTM) hidden layer size
num_layers= 2 # Number of LSTM layers
num_epoch = 5 # The maximum training epochs
learning_rate = 0.0001 # For training
```   
4. **Define Perplexity:**
   - Define perplexity for analysis.

5. **Training the Model:**
   - Train the model using backpropagation.
   - Detach LSTM states at the end of each epoch.
   - Key steps:
      1. Get the data and run the model.
      2. Update the parameters.
      3. Save the loss and compute perplexity on the validation data.
      4. If the perplexity is higher than the previous epoch, return the model.

6. **Evaluate the Model:**
   - Plot perplexity on the evaluation data against the number of epochs.

## Describe Model and Results
The model first starts from an embedding layer.
It is responsible for converting discrete tokens (in this case, there are 20598 unique tokens) into continuous vector representations of size 128.

Then, the LSTM layer takes the 128-dimensional input from the embedding layer and processes it through two LSTM layers, each with 64 hidden units. The batch_first=True argument means that the input and output tensors are expected to have the batch size as their first dimension.

Last followed by the Linear layer. It takes the output of the LSTM layer (which has 64 dimensions) and applies a linear transformation to produce an output of size 20598. This size corresponds to the number of possible output tokens, which is the same as the number of unique tokens in the vocabulary.

For the hyperparameters, I set the maxlen as 268, I initially tried 658, which is the maximum length but it ran forever with terrible perplexity. So I tried the mean, but the perplexity is still terrible. so I followed the suggestion on the piazza and changed it to 268, and the perplexity became normal. These are the examples of the results:
```
maxlen: 268, epoch: 10, perplexity:17 -> 9.7
maxlen: 268, epoch: 5, perplexity: 16 -> 10
maxlen: 73, epoch:5, perplexity: 332 -> 179
```
The rest of the hyperparameters are:
```
embed_dim = 128

hidden_dim = 64

num_layers= 2

num_epoch = 5 # The maximum training epochs, more epochs are not improving

learning_rate = 0.0001
```

The experimental procedure involves data processing by removing punctuations and removing "<unk>" , followed by loading data and connecting with lstm (2 layers). We keep track of the perplexity of the validation data and also the loss during training to optimize the model.


**Results**

The results of validation data on perplexity showed that the perplexity is consistently improving. The length of the text affects a lot of how the perplexity was initialed. i.e. 72 max length resulted in 332 perplexity initially. And changing to 268 resulted in more reasonable numbers.


