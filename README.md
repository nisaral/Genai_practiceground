# The Transformer Architecture

The Transformer is an **encoder-decoder** model designed for sequence-to-sequence tasks. It processes input and output sequences using stacks of identical layers, each leveraging attention and feed-forward networks. Figure 1 in the paper illustrates this structure (reproduced with permission for scholarly use).

## Encoder

* **Structure**: 6 identical layers (N=6), each with two sub-layers:
   1. **Multi-Head Self-Attention**: Computes relationships between all input tokens.
   2. **Feed-Forward Network (FFN)**: Applies a fully connected layer to each position independently.
* **Residual Connections**: Each sub-layer adds its input to its output (x + Sublayer(x)), followed by **layer normalization**.
* **Output**: Produces a continuous representation z of dimension d_model=512 for each input token.

## Decoder

* **Structure**: Also 6 layers, each with three sub-layers:
   1. **Masked Multi-Head Self-Attention**: Attends to previous positions only, ensuring auto-regressive generation (predictions for position i depend only on positions <i).
   2. **Encoder-Decoder Attention**: Queries come from the decoder, keys/values from the encoder output, allowing the decoder to focus on input tokens.
   3. **Feed-Forward Network**: Same as in the encoder.
* **Residual Connections and Normalization**: Applied as in the encoder.
* **Output**: Generates the output sequence one token at a time, using previously generated tokens as input.

## Key Components

### 1. Scaled Dot-Product Attention

* Inputs: Query (Q), Key (K), Value (V) matrices of dimensions d_k and d_v.
* Formula: Attention(Q,K,V)=softmax(QK^T/√d_k)V
* **Scaling by √d_k**: Prevents large dot products from pushing softmax into low-gradient regions.
* Efficiently computed in parallel using matrix operations.

### 2. Multi-Head Attention

* Projects Q, K, V into h=8 subspaces (d_k=d_v=d_model/h=64), applies attention in parallel, then concatenates and projects back.
* Allows the model to capture different aspects of relationships (e.g., syntactic, semantic).
* Formula: MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O where head_i=Attention(QW_i^Q, KW_i^K, VW_i^V).

### 3. Position-wise Feed-Forward Networks

* Applied independently to each position: FFN(x)=max(0,xW_1+b_1)W_2+b_2
* d_model=512 (input/output), d_ff=2048 (inner layer).

### 4. Positional Encoding

* Since there's no recurrence, the model needs to know token positions.
* Uses fixed sine/cosine functions: 
  * PE(pos,2i)=sin(pos/10000^(2i/d_model))
  * PE(pos,2i+1)=cos(pos/10000^(2i/d_model))
* Added to input embeddings, allowing the model to learn relative positions.

### 5. Embeddings and Softmax

* Input/output tokens are embedded into d_model=512 vectors.
* Shared weights between input/output embeddings and the final softmax layer, scaled by √d_model

## Advantages Over RNNs/CNNs

* **Parallelization**: Self-attention processes all tokens simultaneously, unlike RNNs (O(n) sequential steps).
* **Constant Path Length**: Connects any two positions with O(1) operations, vs. O(n) for RNNs or O(log_k(n)) for CNNs (Table 1).
* **Scalability**: Handles long sequences better, though complexity is O(n^2·d) per layer (later optimized in models like Performer).
![image](https://github.com/user-attachments/assets/57f20652-4d16-41ba-8886-086d3692513b)
