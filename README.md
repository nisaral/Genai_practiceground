# Genai_practiceground
## transformer architecture
Transformer models use self-attention mechanisms to weigh the
importance of different words in a sentence, consisting of an encoder to
process input text and a decoder to generate output text, with parallel
processing enabling efficient training.

The Transformer is an encoder-decoder model designed for sequence-to-sequence tasks. It processes input and output sequences using stacks of identical layers, each leveraging attention and feed-forward networks. Figure 1 in the paper illustrates this structure (reproduced with permission for scholarly use).

Encoder
Structure: 6 identical layers (N=6), each with two sub-layers:
Multi-Head Self-Attention: Computes relationships between all input tokens.
Feed-Forward Network (FFN): Applies a fully connected layer to each position independently.
Residual Connections: Each sub-layer adds its input to its output (x + Sublayer(x)), followed by layer normalization.
Output: Produces a continuous representation z of dimension d_model=512 for each input token.
Decoder
Structure: Also 6 layers, each with three sub-layers:
Masked Multi-Head Self-Attention: Attends to previous positions only, ensuring auto-regressive generation (predictions for position i depend only on positions <i).
Encoder-Decoder Attention: Queries come from the decoder, keys/values from the encoder output, allowing the decoder to focus on input tokens.
Feed-Forward Network: Same as in the encoder.
Residual Connections and Normalization: Applied as in the encoder.
Output: Generates the output sequence one token at a time, using previously generated tokens as input.
Key Components
Scaled Dot-Product Attention:
Inputs: Query (Q), Key (K), Value (V) matrices of dimensions d_k and d_v.
Formula:
Attention
(
𝑄
,
𝐾
,
𝑉
)
=
softmax
(
𝑄
𝐾
𝑇
𝑑
𝑘
)
𝑉
Attention(Q,K,V)=softmax( 
d 
k
​
 
​
 
QK 
T
 
​
 )V
Scaling by 
𝑑
𝑘
d 
k
​
 
​
 : Prevents large dot products from pushing softmax into low-gradient regions.
Efficiently computed in parallel using matrix operations.
Multi-Head Attention:
Projects Q, K, V into h=8 subspaces (d_k=d_v=d_model/h=64), applies attention in parallel, then concatenates and projects back.
Allows the model to capture different aspects of relationships (e.g., syntactic, semantic).
Formula:
MultiHead
(
𝑄
,
𝐾
,
𝑉
)
=
Concat
(
head
1
,
.
.
.
,
head
ℎ
)
𝑊
𝑂
MultiHead(Q,K,V)=Concat(head 
1
​
 ,...,head 
h
​
 )W 
O
 
where 
head
𝑖
=
Attention
(
𝑄
𝑊
𝑖
𝑄
,
𝐾
𝑊
𝑖
𝐾
,
𝑉
𝑊
𝑖
𝑉
)
head 
i
​
 =Attention(QW 
i
Q
​
 ,KW 
i
K
​
 ,VW 
i
V
​
 ).
Position-wise Feed-Forward Networks:
Applied independently to each position:
FFN
(
𝑥
)
=
max
⁡
(
0
,
𝑥
𝑊
1
+
𝑏
1
)
𝑊
2
+
𝑏
2
FFN(x)=max(0,xW 
1
​
 +b 
1
​
 )W 
2
​
 +b 
2
​
 
d_model=512 (input/output), d_ff=2048 (inner layer).
Positional Encoding:
Since there’s no recurrence, the model needs to know token positions.
Uses fixed sine/cosine functions:
𝑃
𝐸
(
𝑝
𝑜
𝑠
,
2
𝑖
)
=
sin
⁡
(
𝑝
𝑜
𝑠
1000
0
2
𝑖
/
𝑑
model
)
,
𝑃
𝐸
(
𝑝
𝑜
𝑠
,
2
𝑖
+
1
)
=
cos
⁡
(
𝑝
𝑜
𝑠
1000
0
2
𝑖
/
𝑑
model
)
PE(pos,2i)=sin( 
10000 
2i/d 
model
​
 
 
pos
​
 ),PE(pos,2i+1)=cos( 
10000 
2i/d 
model
​
 
 
pos
​
 )
Added to input embeddings, allowing the model to learn relative positions.
Embeddings and Softmax:
Input/output tokens are embedded into d_model=512 vectors.
Shared weights between input/output embeddings and the final softmax layer, scaled by 
𝑑
model
d 
model
​
 
​
 .
Advantages Over RNNs/CNNs
Parallelization: Self-attention processes all tokens simultaneously, unlike RNNs (O(n) sequential steps).
Constant Path Length: Connects any two positions with O(1) operations, vs. O(n) for RNNs or O(log_k(n)) for CNNs (Table 1).
Scalability: Handles long sequences better, though complexity is O(n^2·d) per layer (later optimized in models like Performer).

![image](https://github.com/user-attachments/assets/57f20652-4d16-41ba-8886-086d3692513b)
