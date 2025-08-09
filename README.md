# GPT2-LSTM
Implemented GPT2 and LSTM from scratch using PyTorch and evaluated the models on a cooking recipe dataset.

Model Architectures
GPT (Transformer)
Architecture: A simplified GPT-like Transformer with rotary positional embeddings.

Parameters:

Embedding Size: 256

Layers: 4

Heads: 4

Block Size: 64

Training:

Optimizer: AdamW with cosine decay scheduler.

Mixed-precision (AMP) support for GPU acceleration.

Pros:

Captures long-range dependencies effectively.

Faster generation once trained.

More coherent output.

LSTM
Architecture: 2-layer character-level LSTM with a linear output layer.

Parameters:

Hidden Size: 256

Layers: 2

Training:

Optimizer: Adam

Same batch size and block length as GPT

Pros:

Simpler to implement.

Still performs reasonably well for small to medium-length sequences.

Limitations:

Struggles with longer dependencies.

Less coherent text generation compared to GPT.
