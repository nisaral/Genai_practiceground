# Genai_practiceground
## transformer architecture
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transformer Architecture</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2980b9;
            margin-top: 25px;
        }
        h3 {
            color: #3498db;
        }
        .component {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 4px 4px 0;
        }
        .formula {
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            font-family: 'Courier New', Courier, monospace;
            overflow-x: auto;
        }
        ul, ol {
            padding-left: 25px;
        }
        li {
            margin-bottom: 8px;
        }
        .highlight {
            font-weight: bold;
            color: #e74c3c;
        }
        .note {
            background-color: #ffffcc;
            padding: 10px;
            border-left: 4px solid #f1c40f;
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <h1>The Transformer Architecture</h1>

    <p>The Transformer is an <strong>encoder-decoder</strong> model designed for sequence-to-sequence tasks. It processes input and output sequences using stacks of identical layers, each leveraging attention and feed-forward networks. Figure 1 in the paper illustrates this structure (reproduced with permission for scholarly use).</p>

    <h2>Encoder</h2>
    <div class="component">
        <ul>
            <li><strong>Structure</strong>: 6 identical layers (N=6), each with two sub-layers:
                <ol>
                    <li><strong>Multi-Head Self-Attention</strong>: Computes relationships between all input tokens.</li>
                    <li><strong>Feed-Forward Network (FFN)</strong>: Applies a fully connected layer to each position independently.</li>
                </ol>
            </li>
            <li><strong>Residual Connections</strong>: Each sub-layer adds its input to its output (x + Sublayer(x)), followed by <strong>layer normalization</strong>.</li>
            <li><strong>Output</strong>: Produces a continuous representation z of dimension d_model=512 for each input token.</li>
        </ul>
    </div>

    <h2>Decoder</h2>
    <div class="component">
        <ul>
            <li><strong>Structure</strong>: Also 6 layers, each with three sub-layers:
                <ol>
                    <li><strong>Masked Multi-Head Self-Attention</strong>: Attends to previous positions only, ensuring auto-regressive generation (predictions for position i depend only on positions &lt;i).</li>
                    <li><strong>Encoder-Decoder Attention</strong>: Queries come from the decoder, keys/values from the encoder output, allowing the decoder to focus on input tokens.</li>
                    <li><strong>Feed-Forward Network</strong>: Same as in the encoder.</li>
                </ol>
            </li>
            <li><strong>Residual Connections and Normalization</strong>: Applied as in the encoder.</li>
            <li><strong>Output</strong>: Generates the output sequence one token at a time, using previously generated tokens as input.</li>
        </ul>
    </div>

    <h2>Key Components</h2>

    <h3>1. Scaled Dot-Product Attention</h3>
    <div class="component">
        <ul>
            <li>Inputs: Query (Q), Key (K), Value (V) matrices of dimensions d_k and d_v.</li>
            <li>Formula: 
                <div class="formula">
                    Attention(Q, K, V) = softmax(QK<sup>T</sup>/√d<sub>k</sub>)V
                </div>
            </li>
            <li><strong>Scaling by √d<sub>k</sub></strong>: Prevents large dot products from pushing softmax into low-gradient regions.</li>
            <li>Efficiently computed in parallel using matrix operations.</li>
        </ul>
    </div>

    <h3>2. Multi-Head Attention</h3>
    <div class="component">
        <ul>
            <li>Projects Q, K, V into h=8 subspaces (d_k=d_v=d_model/h=64), applies attention in parallel, then concatenates and projects back.</li>
            <li>Allows the model to capture different aspects of relationships (e.g., syntactic, semantic).</li>
            <li>Formula: 
                <div class="formula">
                    MultiHead(Q, K, V) = Concat(head<sub>1</sub>, ..., head<sub>h</sub>)W<sup>O</sup><br>
                    where head<sub>i</sub> = Attention(QW<sub>i</sub><sup>Q</sup>, KW<sub>i</sub><sup>K</sup>, VW<sub>i</sub><sup>V</sup>)
                </div>
            </li>
        </ul>
    </div>

    <h3>3. Position-wise Feed-Forward Networks</h3>
    <div class="component">
        <ul>
            <li>Applied independently to each position: 
                <div class="formula">
                    FFN(x) = max(0, xW<sub>1</sub> + b<sub>1</sub>)W<sub>2</sub> + b<sub>2</sub>
                </div>
            </li>
            <li>d_model=512 (input/output), d_ff=2048 (inner layer).</li>
        </ul>
    </div>

    <h3>4. Positional Encoding</h3>
    <div class="component">
        <ul>
            <li>Since there's no recurrence, the model needs to know token positions.</li>
            <li>Uses fixed sine/cosine functions: 
                <div class="formula">
                    PE(pos, 2i) = sin(pos/10000<sup>2i/d<sub>model</sub></sup>)<br>
                    PE(pos, 2i+1) = cos(pos/10000<sup>2i/d<sub>model</sub></sup>)
                </div>
            </li>
            <li>Added to input embeddings, allowing the model to learn relative positions.</li>
        </ul>
    </div>

    <h3>5. Embeddings and Softmax</h3>
    <div class="component">
        <ul>
            <li>Input/output tokens are embedded into d_model=512 vectors.</li>
            <li>Shared weights between input/output embeddings and the final softmax layer, scaled by √d<sub>model</sub>.</li>
        </ul>
    </div>

    <h2>Advantages Over RNNs/CNNs</h2>
    <div class="component">
        <ul>
            <li><strong>Parallelization</strong>: Self-attention processes all tokens simultaneously, unlike RNNs (O(n) sequential steps).</li>
            <li><strong>Constant Path Length</strong>: Connects any two positions with O(1) operations, vs. O(n) for RNNs or O(log<sub>k</sub>(n)) for CNNs (Table 1).</li>
            <li><strong>Scalability</strong>: Handles long sequences better, though complexity is O(n²·d) per layer (later optimized in models like Performer).</li>
        </ul>
    </div>
</body>
</html>
![image](https://github.com/user-attachments/assets/57f20652-4d16-41ba-8886-086d3692513b)
