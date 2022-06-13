# 论文报告

**A Survey on Vision Transformer**

大纲：讲解transformer的各种应用

## FORMULATION OF TRANSFORMER：

Transformer was first used in the field of natural language processing (NLP) on machine translation tasks.

it consists of an encoder and a decoder with several transformer blocks of the same architecture. The encoder gener- ates encodings of inputs, while the decoder takes all the encodings and using their incorporated contextual information to generate the output sequence. 

Each transformer block is composed of a multi-head attention layer, a feed-forward neural network, shortcut connection and layer normalization. 

### self-attention

 the input vector is first transformed into three different vectors: the query vector q, the key vector k and the value vector v with dimension dq = dk = dv = dmodel = 512. Vectors derived from different inputs are then packed together into three different matrices, namely, Q, K and V. Subsequently, the attention function between different input vectors is calculated as follows (and shown in Figure 3 left):
- Step 1: Compute scores between different input vectors with S = Q · K⊤;
- Step 2: Normalize the scores for the stability of gradient √
with Sn = S/ dk;
- Step 3: Translate the scores into probabilities with softmax
function P = softmax(Sn);
- Step 4: Obtain the weighted value matrix with Z = V · P.

The process can be unified into a single function:

$$Attention(Q, K, V) = softmax(\frac{Q✖️K^T}{\sqrt{d_k}})*V$$

The encoder-decoder attention layer in the decoder module is similar to the self-attention layer in the encoder module with the following exceptions: The key matrix K and value matrix V are derived from the encoder module, and the query matrix Q is derived from the previous layer.

Note that the preceding process is invariant to the position of each word, meaning that the self-attention layer lacks the ability to capture the positional information of words in a sentence.

#### positional encoding is needed

Specifically, the position is encoded with the following equations:

$$PE(pos,2i) = sin(\frac{pos}{10000^{\frac{2i}{d_model}}})$$
$$PE(pos,2i+1) = cos(\frac{pos}{10000^{\frac{2i}{d_model}}})$$

In which pos denotes the position of the word in a sentence, and i represents the current dimension of the positional encoding. In this way, each element of the positional encoding corresponds to a sinusoid, and it allows the transformer model to learn to attend by relative positions and extrapolate to longer sequence lengths during inference. 