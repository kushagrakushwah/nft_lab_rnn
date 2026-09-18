# Deep Learning Master Guide: RNN, LSTM, GRU & Sentiment Analysis

A complete, term-by-term theoretical guide and full source code breakdown for mastering sequential modeling, Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU) applied to NLP Sentiment Analysis.

---

## Table of Contents
1. [Core NLP & Sequential Modeling Terminology](#1-core-nlp--sequential-modeling-terminology)
2. [Vanilla Recurrent Neural Networks (RNN)](#2-vanilla-recurrent-neural-networks-rnn)
3. [Long Short-Term Memory Networks (LSTM)](#3-long-short-term-memory-networks-lstm)
4. [Gated Recurrent Units (GRU)](#4-gated-recurrent-units-gru)
5. [Architecture Comparison Matrix](#5-architecture-comparison-matrix)
6. [Complete Code Walkthrough & Line-by-Line Breakdown](#6-complete-code-walkthrough--line-by-line-breakdown)
   - [dataset.py](#datasetpy)
   - [rnn_model.py](#rnn_modelpy)
   - [lstm_model.py](#lstm_modelpy)
   - [gru_model.py](#gru_modelpy)
   - [train_and_compare.py](#train_and_comparepy)
   - [predict_sentences.py](#predict_sentencespy)
   - [sentiment_analysis.py](#sentiment_analysispy)

---

## 1. Core NLP & Sequential Modeling Terminology

Before diving into neural network cells, you must understand how text is transformed from raw strings into numerical matrices:

### 1.1 Tokenization
* **Definition**: The process of breaking a continuous stream of text into discrete atomic units called **tokens** (usually words or subwords).
* **Example**: `"The movie was great!"` $\to$ `["the", "movie", "was", "great"]`.
* **Why it matters**: Neural networks cannot read raw text; they require discrete units to assign numerical IDs.

### 1.2 Vocabulary (`word2idx` & `idx2word`)
* **Definition**: The complete set of unique tokens discovered across your training corpus.
* **`word2idx`**: A hash map / dictionary converting a string token to a unique integer ID (e.g., `{"the": 2, "movie": 3, ...}`).
* **`idx2word`**: The reverse lookup dictionary converting an integer ID back into a string word.

### 1.3 Special Tokens (`<PAD>` and `<UNK>`)
* **`<PAD>` (Padding Token, Index 0)**: Sentences in a mini-batch have variable lengths. To stack them into a rectangular matrix $[B, T]$, shorter sentences are filled with `<PAD>` zeros at the end.
* **`<UNK>` (Unknown Token, Index 1)**: Any word encountered during testing that was not present in the training vocabulary is mapped to `<UNK>` to prevent out-of-vocabulary crashes.

### 1.4 Word Embeddings
* **Definition**: Dense, continuous vector representations of words in $\mathbb{R}^{d}$ (e.g., $d = 64$).
* **Why not One-Hot Encoding?**: A one-hot vector for a vocabulary of size 10,000 has length 10,000, is 99.99% sparse zeros, and gives zero semantic similarity between related words like `"fantastic"` and `"excellent"`.
* **PyTorch Module**: `nn.Embedding(vocab_size, embedding_dim)`. Acts as a trainable lookup table where each row is the learned continuous vector for a word.

### 1.5 Many-to-One Sequence Classification
* **Definition**: A sequence of $T$ input tokens $x_1, x_2, \dots, x_T$ is fed into a recurrent network step-by-step, but only the **final hidden state** $h_T$ (which encapsulates the accumulated meaning of the entire sentence) is passed to a classification layer.

### 1.6 Binary Cross-Entropy Loss with Logits (`BCEWithLogitsLoss`)
* **Logits**: Raw, unnormalized real-numbered outputs from a linear layer ($z \in (-\infty, +\infty)$).
* **Sigmoid Activation**: $\sigma(z) = \frac{1}{1 + e^{-z}}$, squashing logits into probabilities $p \in (0, 1)$.
* **BCE Loss Formula**:
  $$\mathcal{L} = - \left[ y \log \sigma(z) + (1 - y) \log (1 - \sigma(z)) \right]$$
* **Why numerically stable**: PyTorch's `BCEWithLogitsLoss` combines the Sigmoid and Log loss into a single mathematical step using the log-sum-exp trick to prevent numerical overflow and underflow.

---

## 2. Vanilla Recurrent Neural Networks (RNN)

### 2.1 The Concept
Traditional Feedforward Networks process each input independently with no memory. An RNN processes a sequence one element at a time, maintaining an internal memory vector called the **Hidden State** ($h_t$).

### 2.2 Mathematical Equation
At time step $t$, given input embedding $x_t \in \mathbb{R}^d$ and previous hidden state $h_{t-1} \in \mathbb{R}^h$:

$$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$

Where:
* $W_{ih} \in \mathbb{R}^{h \times d}$: Weight matrix projecting the input word vector into the hidden space.
* $W_{hh} \in \mathbb{R}^{h \times h}$: Recurrent weight matrix projecting the previous hidden state forward.
* $b_{ih}, b_{hh} \in \mathbb{R}^h$: Learnable bias vectors.
* $\tanh$: Hyperbolic tangent activation function, constraining activations between $-1$ and $+1$.

### 2.3 Unfolding Over Time
```
Time Step 1:         Time Step 2:                  Time Step T:
   x_1                  x_2                           x_T
    ↓                    ↓                             ↓
[RNN Cell]  --h_1-->  [RNN Cell]  -- ... --h_{T-1}--> [RNN Cell] --h_T--> [Linear Head] --> Sentiment
```

### 2.4 Backpropagation Through Time (BPTT) & Vanishing Gradients
To update weights, errors must backpropagate backwards from step $T$ to step 1. By the chain rule:

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{k=2}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

Where:
$$\frac{\partial h_k}{\partial h_{k-1}} = \text{diag}(1 - \tanh^2(\cdot)) \cdot W_{hh}^T$$

**The Fatal Flaw**:
1. $\tanh'(z) = 1 - \tanh^2(z) \le 1$ (with a maximum of 1 at 0, and rapidly decaying toward 0).
2. If the singular values of $W_{hh}$ are $< 1$, multiplying by $W_{hh}^T$ repeatedly across 20 time steps causes gradients to decay exponentially to 0 ($0.8^{20} \approx 0.011$).
3. **Result**: Vanilla RNN cannot learn dependencies between words that are separated by more than 8–10 tokens (e.g., negations like *"not ... good"* get forgotten).

---

## 3. Long Short-Term Memory Networks (LSTM)

### 3.1 The Motivation
Invented by Hochreiter & Schmidhuber (1997) specifically to solve the vanishing gradient problem.

### 3.2 The Core Secret: Cell State ($C_t$) vs Hidden State ($h_t$)
* **Cell State ($C_t$)**: Acts like a conveyor belt running straight through the entire chain with only minor linear interactions. It is the network's **long-term memory**.
* **Hidden State ($h_t$)**: The filtered, working memory exposed to the outside world at the current step.

### 3.3 The Three Multiplicative Gates (Term-by-Term)

#### Gate 1: Forget Gate ($f_t$)
* **Purpose**: Looks at current word $x_t$ and previous state $h_{t-1}$, and decides what percentage of old cell memory $C_{t-1}$ to throw away.
* **Equation**:
  $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
* **Interpretation**: If $f_t = 0$, completely wipe out memory; if $f_t = 1$, preserve 100% of memory.

#### Gate 2: Input Gate ($i_t$) & Candidate Cell State ($\tilde{C}_t$)
* **Purpose**: Decides what brand new information to store in the cell state.
* **Equations**:
  $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(How much to write)}$$
  $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{(What new candidate concepts to write)}$$

#### State Update: Additive Cell State Update ($C_t$)
* **Purpose**: Updates old memory $C_{t-1}$ into new memory $C_t$:
  $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
* **Why this solves Vanishing Gradients**: Because the update is **additive** (+), the derivative $\frac{\partial C_t}{\partial C_{t-1}} = f_t$. When the forget gate is open ($f_t \approx 1$), gradient signals flow across hundreds of time steps with zero attenuation!

#### Gate 3: Output Gate ($o_t$) & Hidden State ($h_t$)
* **Purpose**: Decides what parts of the updated cell state to output as the visible hidden state $h_t$.
* **Equations**:
  $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
  $$h_t = o_t \odot \tanh(C_t)$$

### 3.4 Bidirectional LSTM (BiLSTM)
* Instead of reading only left-to-right, a BiLSTM runs two independent LSTMs:
  1. Forward LSTM ($\vec{h}_t$): Reads words $1 \to T$ (past context).
  2. Backward LSTM ($\overleftarrow{h}_t$): Reads words $T \to 1$ (future context).
* Output hidden state is concatenated: $h_t = [\vec{h}_t \,;\, \overleftarrow{h}_t]$, giving full contextual awareness of both preceding and subsequent words.

---

## 4. Gated Recurrent Units (GRU)

### 4.1 The Motivation
Introduced by Cho et al. (2014), GRU simplifies LSTM by removing the separate Cell State ($C_t$) and reducing the number of gates from 3 down to 2, making training faster and computationally cheaper.

### 4.2 The Two Gates (Term-by-Term)

#### Gate 1: Reset Gate ($r_t$)
* **Purpose**: Decides how much of the past hidden state $h_{t-1}$ to forget when computing the new candidate state.
* **Equation**:
  $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

#### Gate 2: Update Gate ($z_t$)
* **Purpose**: Acts simultaneously as both the forget gate and input gate in LSTM. It determines the balance between old state and new state.
* **Equation**:
  $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

#### Candidate Hidden State ($\tilde{h}_t$)
* **Purpose**: Proposes candidate new features, applying the reset gate to damp out irrelevant past history:
  $$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

#### Final Hidden State Interpolation ($h_t$)
* **Purpose**: Smooth linear blend between the previous state and candidate state:
  $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
* **Intuition**: If $z_t \approx 1$, the unit completely adopts the new candidate memory; if $z_t \approx 0$, it ignores the current word and copies forward the previous hidden state unchanged.

---

## 5. Architecture Comparison Matrix

| Property | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| **Number of Gates** | 0 | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **Internal States** | Hidden state $h_t$ | Cell state $C_t$ + Hidden state $h_t$ | Hidden state $h_t$ |
| **Parameter Count** | $\mathcal{O}(d \cdot h + h^2)$ | $4 \times \mathcal{O}(d \cdot h + h^2)$ | $3 \times \mathcal{O}(d \cdot h + h^2)$ |
| **Parameters vs LSTM** | ~25% | 100% (Baseline) | ~75% (25% fewer parameters) |
| **Vanishing Gradient** | High risk | Highly resistant | Highly resistant |
| **Training Speed** | Fastest | Slower | Fast (20-30% faster than LSTM) |
| **Primary Use Case** | Simple baselines | Long complex sequences | Medium datasets, faster inference |

---

## 6. Complete Code Walkthrough & Line-by-Line Breakdown

### `dataset.py`

#### What it does:
Defines the sentiment data, implements word tokenization, manages the string-to-integer vocabulary, and pads variable-length sequences into uniform tensors for mini-batching.

#### Complete Annotated Source:
```python
import re
import json
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader

# 1. RAW DATASET: Labeled pairs of (Sentence, Label) where 1 = Positive, 0 = Negative
RAW_DATA: List[Tuple[str, int]] = [
    ("I absolutely love this movie, it was a breathtaking masterpiece!", 1),
    ("This product is a total waste of money, stopped working on day two.", 0),
    ("Not bad at all, actually quite pleasant and well made.", 1),
    ("It was not good at all, completely failed my expectations.", 0),
    # ... (70 comprehensive labeled sentences)
]

class Vocabulary:
    """Manages word-to-integer mappings and special tokens."""
    PAD_TOKEN = "<PAD>"  # Token ID 0: Used to pad short sequences
    UNK_TOKEN = "<UNK>"  # Token ID 1: Used for unknown words
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word: Dict[int, str] = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}

    def build_vocab(self, sentences: List[str], min_freq: int = 1):
        """Counts word frequencies and assigns unique integer IDs."""
        freq: Dict[str, int] = {}
        for sent in sentences:
            tokens = self.tokenize(sent)
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1

        idx = len(self.word2idx)
        for token, count in freq.items():
            if count >= min_freq and token not in self.word2idx:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercases text and extracts words using regular expressions."""
        clean_text = text.lower()
        return re.findall(r"\b\w+(?:'\w+)?\b", clean_text)

    def text_to_indices(self, text: str) -> List[int]:
        """Converts raw sentence string into a list of vocabulary token IDs."""
        tokens = self.tokenize(text)
        return [self.word2idx.get(t, self.UNK_IDX) for t in tokens]

    def indices_to_text(self, indices: List[int]) -> str:
        """Converts integer token IDs back to human-readable text."""
        return " ".join([self.idx2word.get(i, self.UNK_TOKEN) for i in indices if i != self.PAD_IDX])

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx}, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        vocab = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab


class SentimentDataset(Dataset):
    """PyTorch Dataset returning numerical sequences, sentiment labels, and sentence lengths."""
    def __init__(self, data: List[Tuple[str, int]], vocab: Vocabulary):
        self.sentences = [s for s, _ in data]
        self.labels = [lbl for _, lbl in data]
        self.vocab = vocab
        self.encoded_data = [self.vocab.text_to_indices(s) for s in self.sentences]

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[List[int], int, int]:
        seq = self.encoded_data[idx]
        return seq, self.labels[idx], len(seq)


def pad_collate_fn(batch: List[Tuple[List[int], int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates a list of variable-length samples into a padded rectangular batch tensor."""
    sequences, labels, lengths = zip(*batch)
    max_len = max(lengths)
    
    padded_seqs = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return padded_seqs, labels_tensor, lengths_tensor


def get_data_loaders(batch_size: int = 8, split_ratio: float = 0.8, seed: int = 42):
    """Splits dataset into 80% train / 20% validation and builds PyTorch DataLoaders."""
    import random
    random.seed(seed)
    shuffled = RAW_DATA.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * split_ratio)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]

    vocab = Vocabulary()
    vocab.build_vocab([s for s, _ in train_data], min_freq=1)

    train_dataset = SentimentDataset(train_data, vocab)
    val_dataset = SentimentDataset(val_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    return train_loader, val_loader, vocab
```

---

### `rnn_model.py`

#### What it does:
Builds the Vanilla RNN architecture using `nn.RNN`, passes sequences through word embeddings, extracts the hidden state at the final valid token index, and projects to binary sentiment logits.

#### Key Code Snippet & Mechanics:
```python
class VanillaRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1, n_layers=1, dropout=0.2, pad_idx=0):
        super().__init__()
        self.model_type = "RNN"
        self.hidden_dim = hidden_dim
        
        # 1. Lookup table mapping token IDs to 64-dimensional dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # 2. Vanilla Elman RNN: h_t = tanh(W_ih * x_t + W_hh * h_{t-1})
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        # 3. Output classification projection from hidden space (64) to single logit (1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, lengths=None):
        # embedded shape: [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(text))
        
        # rnn_out shape: [batch_size, seq_len, hidden_dim]
        rnn_out, hidden = self.rnn(embedded)

        # Extract the hidden state at the exact final unpadded token of each sentence
        if lengths is not None:
            batch_size = text.size(0)
            idx = (lengths - 1).view(-1, 1).expand(batch_size, self.hidden_dim).unsqueeze(1)
            last_hidden = rnn_out.gather(1, idx).squeeze(1)
        else:
            last_hidden = hidden[-1]

        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped).squeeze(1)
        return logits
```

---

### `lstm_model.py`

#### What it does:
Implements Bidirectional LSTM. At each time step, calculates forget, input, and output gates along with the additive cell state update. Concatenates forward and backward hidden states for context-aware classification.

#### Key Code Snippet & Mechanics:
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1, n_layers=1, bidirectional=True, dropout=0.3, pad_idx=0):
        super().__init__()
        self.model_type = "LSTM"
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM includes C_t (cell state) and 3 gates (f_t, i_t, o_t)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        # Bidirectional concatenates forward (64) + backward (64) = 128 inputs to fc
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)

    def forward(self, text, lengths=None):
        embedded = self.dropout(self.embedding(text))
        # lstm returns output along with (hidden, cell) tuple
        lstm_out, (hidden, cell) = self.lstm(embedded)

        if self.bidirectional:
            # Concatenate the final forward hidden state and backward hidden state
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            last_hidden = hidden[-1]

        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped).squeeze(1)
        return logits
```

---

### `gru_model.py`

#### What it does:
Implements Bidirectional GRU. Merges cell state into hidden state, uses reset and update gates to control information flow, achieving ~25% parameter reduction compared to LSTM.

#### Key Code Snippet & Mechanics:
```python
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1, n_layers=1, bidirectional=True, dropout=0.3, pad_idx=0):
        super().__init__()
        self.model_type = "GRU"
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # GRU replaces 3 gates of LSTM with reset gate (r_t) and update gate (z_t)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)

    def forward(self, text, lengths=None):
        embedded = self.dropout(self.embedding(text))
        gru_out, hidden = self.gru(embedded)

        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            last_hidden = hidden[-1]

        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped).squeeze(1)
        return logits
```

---

### `train_and_compare.py`

#### What it does:
Trains all 3 models under identical conditions (same random seed, learning rate, embedding dimensions, epochs), tracks parameter counts, training times, inference latencies, and prints a comparative table.

#### Workflow Summary:
1. Fix random seed: `torch.manual_seed(42)`.
2. Load and pad training/validation datasets.
3. Instantiate `VanillaRNNClassifier`, `LSTMClassifier`, and `GRUClassifier`.
4. Run 25 training epochs using `BCEWithLogitsLoss` and `torch.optim.Adam`.
5. Clip gradients (`clip_grad_norm_`) to prevent exploding gradients.
6. Evaluate validation accuracy and measure inference latency over 100 iterations.
7. Save model checkpoints (`rnn_model.pt`, `lstm_model.pt`, `gru_model.pt`) and `vocab.json`.

---

### `predict_sentences.py`

#### What it does:
The dedicated multi-sentence inference suite. Evaluates 16 sentences covering clear positive/negative, negations, and contrastive phrases across all 3 models, showing side-by-side probabilities and predictions.

#### Key Code Snippet:
```python
def predict_sentence_all_models(sentence: str, models: Dict[str, nn.Module], vocab: Vocabulary, device: torch.device):
    """Converts a sentence into token IDs and queries RNN, LSTM, and GRU simultaneously."""
    indices = vocab.text_to_indices(sentence)
    tensor = torch.tensor([indices], dtype=torch.long, device=device)
    length = torch.tensor([len(indices)], dtype=torch.long, device=device)

    results = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            logit = model(tensor, length)
            prob = torch.sigmoid(logit).item()
            label = "Positive" if prob >= 0.5 else "Negative"
            results[name] = (label, prob)
    return results
```
