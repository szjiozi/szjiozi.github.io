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
Attention(Q, K, V) = softmax((Q✖️K T)/根号dk)✖️V
