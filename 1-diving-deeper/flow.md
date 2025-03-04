# Transformer Encoder Block Anatomy

## Data Flow Diagram

```mermaid
    A[Input Embeddings + Positional Encoding] --> B[Multi-Head Self-Attention]
    B[Multi-Head Self-Attention] --> C[Add & Norm]
    A[Input Embeddings + Positional Encoding] --> C[Add & Norm]
    C[Add & Norm] --> D[Feed Forward Network]
    D[Feed Forward Network] --> E[Add & Norm]
    C[Add & Norm] --> E[Add & Norm]
    E[Add & Norm] --> F[Output Embeddings]
    F[Output Embeddings] = Input to the decoder block
```

## Step-by-Step Process

### 1. Multi-Head Self-Attention

```mermaid
    A[Input] --> B[Linear Projections]
    B[Linear Projections] --> C1[Head 1]
    B[Linear Projections] --> C2[Head 2]
    B[Linear Projections] --> C3[...]
    B[Linear Projections] --> Cn[Head N]
    C1[Head 1] --> D[Compute Attention]
    C2[Head 2] --> D[Compute Attention]
    C3[Head 3] --> D[Compute Attention]
    Cn[Head N] --> D[Compute Attention]
    D[Compute Attention] --> E[Concatenate]
    E[Concatenate] --> F[Final Linear Projection]
```

**Key Operations**:
1. Split embedding into `num_heads` different subspaces
2. Each head computes scaled dot-product attention independently
3. Concatenate all head outputs
4. Linear projection to original dimension

### 2. First Add & Norm (Residual Connection + Layer Normalization for the output of the multi-head attention)

```mermaid
    A[Attention Output] --> B[Add]
    C[Original Input] --> B[Add]
    B[Add] --> D[LayerNorm]
    D[LayerNorm] --> E[Output]
```

**Mathematically**:
```
LayerNorm(x + Sublayer(x))
```

### 3. Position-wise Feed Forward Network

```mermaid
    A[Input] --> B[Linear: d_model → d_ff]
    B[Linear: d_model → d_ff] --> C[ReLU]
    C --> D[Linear: d_ff → d_model]
```

**Dimension Example**:
`512 → 2048 → 512`

### 4. Final Add & Norm (Residual Connection + Layer Normalization for the output of the FFN)

```mermaid
    A[FFN Output] --> B[Add]
    C[Previous Output] --> B
    B --> D[LayerNorm]
    D --> E[Final Output]
```

## Critical Properties Table

| Property | Purpose | Implementation Note |
|----------|---------|---------------------|
| Residual Connections | Prevent vanishing gradients, enable deep networks | Simple element-wise addition |
| Layer Normalization | Stabilize training, reduce sensitivity to initialization | Normalize across feature dimension |
| Multi-Head Attention | Capture different types of relationships simultaneously | Parallel attention mechanisms |
| Position-wise FFN | Add non-linearity and transformation capacity | Applied independently per position |

## Visualized Mathematical Flow

<img src="../assets/encoder-flow-diagram.png" alt="Transformer Encoder Block" width="500">

## Concrete Example Walkthrough

(for batch size=2, seq_len=4, d_model=512):

1. **Multi-Head Attention**:
```zsh
Input: [2,4,512] → Split into 8 heads → [2,8,4,64]
       ▼
Self-Attention per head → [2,8,4,64]
       ▼
Concatenate → [2,4,512] → Linear → [2,4,512]
```

2. **Add & Norm**:
```zsh
Original: [2,4,512]  
Attention Output: [2,4,512]
Add → [2,4,512] → LayerNorm → [2,4,512]
```

3. **FFN**:
```zsh
[2,4,512] → Linear → [2,4,2048] → ReLU → Linear → [2,4,512]
```

4. **Final Add & Norm**:
```zsh
FFN Output: [2,4,512]  
Previous: [2,4,512]
Add → [2,4,512] → LayerNorm → Final Output
```

**Code Mapping**:
```python
# Full encoder block sequence:
input = encoded_embeddings  # [2,4,512]
attn_output = multi_head_attention(input)  # Step 1
norm1 = norm(input + attn_output)  # Step 2
ffn_output = ffn(norm1)  # Step 3
final_output = norm(norm1 + ffn_output)  # Step 4
```




