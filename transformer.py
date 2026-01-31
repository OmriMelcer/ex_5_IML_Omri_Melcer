import math
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from dataset import DataHandler
import matplotlib.pyplot as plt
import os


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))




class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        #### YOUR CODE HERE ####
        # TIP: 
        self.input_projection = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.output_projection = nn.Linear(self.n_embd, self.n_embd)
        # It is common practive to initialze a single Linear layer to map each token to its query, key, and value, i.e. nn.Linear(self.n_embd, 3 * self.n_embd)
        # After applying the linear layer on a token embedding you can split the layer's output to key, query, and value

        # The output key/query/value is of dimension n_embd, in practice this includes the embeddings for all heads, 
        # therefore, embedding = [embd_1, embd_2, .. embd_nheads]. You can rearange as you please in the forward pass.
        
        

    def forward(self, x):
        #### YOUR CODE HERE ####
        # Compute queries, keys, and values. Expected shape [batch_size, n_heads, sequence_length n_embd/n_head]
        after_input_proj = self.input_projection(x) # shape [batch_size, sequence_length, 3*n_embd]
        Q, K, V = torch.chunk(after_input_proj, 3, dim=-1)  # Each of shape [batch_size, sequence_length, n_embd]
        Q = Q.view(Q.size(0), Q.size(1), self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # shape [batch_size, n_heads, sequence_length, n_embd/n_head]
        K = K.view(K.size(0), K.size(1), self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # shape [batch_size, n_heads, sequence_length, n_embd/n_head]
        V = V.view(V.size(0), V.size(1), self.n_head, self.n_embd // self.n_head).transpose(1, 2)  # shape [batch_size, n_heads, sequence_length, n_embd/n_head]
        # Compute normalized attention matrix (Q@K.T)/sqrt(d_k), Expected shape [batch_size, n_heads, sequence_length, sequence_length]
        d_k = self.n_embd // self.n_head
        # Mask, this is casual self-attention, you need to mask the score of each token with the tokens that come after it in the sequence
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # shape [batch_size, n_heads, sequence_length, sequence_length]
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        # NOTE: the dimension d_k refers to the embedding dimension of the keys which is n_embd/num_heads 
        attention_weights = torch.softmax(scores, dim=-1)  # shape [batch_size, n_heads, sequence_length, sequence_length]
        attention_output = torch.matmul(attention_weights, V)  # shape [batch_size, n_heads, sequence_length, n_embd/n_head]
        attention_output = attention_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.n_embd)
        return self.output_projection(attention_output)  # shape [batch_size, sequence_length, n_embd]
        

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """


    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, self.n_embd),
            wpe = nn.Embedding(block_size, self.n_embd),            
            h = nn.ModuleList([Block(n_head, n_embd, block_size) for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)



    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits





def train_model(
        train_path,
        test_path=None,
        model=None,
        block_size=10,
        n_layer=3,
        n_head=3,
        n_embd=48,
        learning_rate=3e-4,
        batch_size=64,
        epochs=10,
        log_file='transformer_training.log',
        model_save_path='gpt_model.pt'
):
    # Open log file
    log_f = open(log_file, 'w')
    log_f.write(f"Training GPT Model\n")
    log_f.write(f"Parameters: block_size={block_size}, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}\n")
    log_f.write(f"learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}\n")
    log_f.write("="*80 + "\n\n")

    data_handler = DataHandler(train_path, test_path, block_size)

    device = torch.device('cpu')
    if (torch.cuda.is_available()):
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    vocab_size = data_handler.get_vocab_size()
    if model is None:
        model = GPT(n_layer, n_head, n_embd, vocab_size, block_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    log_f.write(f'Using device: {device}\n')
    log_f.write(f'Vocabulary size: {vocab_size}\n\n')

    trainset = data_handler.get_dataset('train')
    testset = data_handler.get_dataset('test')

    # setup the dataloader
    train_loader = DataLoader(
        trainset,
        sampler=torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=int(1e5)),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
    )
    if testset:
        test_loader = DataLoader(
            testset,
            sampler=torch.utils.data.RandomSampler(testset, replacement=False, num_samples=int(1e4)),
            shuffle=False,
            pin_memory=True,
            batch_size=batch_size,
        )

    # Tracking metrics
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for ep in range(epochs):
        log_f.write(f"\n{'='*80}\n")
        log_f.write(f"Epoch {ep+1}/{epochs}\n")
        log_f.write(f"{'='*80}\n")

        # Training phase
        model.train()
        train_loss_accum = 0.0
        train_correct = 0.0
        train_total = 0

        for i, batch in enumerate(tqdm(train_loader)):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            # Track training metrics
            train_loss_accum += loss.item()
            predictions = torch.argmax(outputs, dim=-1)
            train_correct += (predictions == targets).float().sum().item()
            train_total += targets.numel()

        train_loss = train_loss_accum / len(train_loader)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        log_f.write(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}\n")

        # Evaluation phase
        with torch.no_grad():
            model.eval()
            test_loss_accum = 0.0
            test_correct = 0.0
            test_total = 0

            for i, batch in enumerate(tqdm(test_loader)):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Calculate metrics
                predictions = torch.argmax(outputs, dim=-1)
                test_correct += (predictions == targets).float().sum().item()
                test_total += targets.numel()

                loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
                test_loss_accum += loss.item()

            test_loss = test_loss_accum / len(test_loader)
            test_accuracy = test_correct / test_total
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            log_f.write(f"Testing  - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}\n\n")

            # Generate sentences (regular sampling)
            log_f.write("Generated sentences (regular sampling):\n")
            for sample_idx in range(3):
                sentence = "the "
                new_sentence = sentence
                for char_idx in range(20):
                    tokens = torch.tensor(data_handler.encoder(new_sentence[-block_size:]))[None]
                    tokens = tokens.to(device)
                    outputs = model(tokens)
                    next_token_logits = outputs[0, -1, :]
                    probabilities = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1).item()
                    next_char = data_handler.decoder([next_token])
                    new_sentence += next_char
                log_f.write(f"  {sample_idx+1}. {new_sentence}\n")

            # Generate sentences (top-k sampling with k=5)
            log_f.write("\nGenerated sentences (top-k sampling, k=5):\n")
            for sample_idx in range(3):
                sentence = "the "
                new_sentence = sentence
                for char_idx in range(20):
                    tokens = torch.tensor(data_handler.encoder(new_sentence[-block_size:]))[None]
                    tokens = tokens.to(device)
                    outputs = model(tokens)
                    next_token_logits = outputs[0, -1, :]
                    probabilities = torch.softmax(next_token_logits, dim=-1)

                    # Get top-k values and indices
                    top_k_probs, top_k_indices = torch.topk(probabilities, k=5)
                    # Renormalize
                    top_k_probs = top_k_probs / torch.sum(top_k_probs)
                    # Sample from top-k
                    sampled_idx = torch.multinomial(top_k_probs, num_samples=1).item()
                    next_token = top_k_indices[sampled_idx].item()
                    next_char = data_handler.decoder([next_token])
                    new_sentence += next_char
                log_f.write(f"  {sample_idx+1}. {new_sentence}\n")

            log_f.flush()

    # Save model
    log_f.write(f"\n{'='*80}\n")
    log_f.write(f"Saving model to {model_save_path}\n")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }, model_save_path)
    log_f.write("Model saved successfully!\n")

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, epochs + 1)

    # Loss plot
    ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss', marker='o')
    ax1.plot(epochs_range, test_losses, 'r-', label='Test Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss vs. Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs_range, train_accuracies, 'b-', label='Train Accuracy', marker='o')
    ax2.plot(epochs_range, test_accuracies, 'r-', label='Test Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Test Accuracy vs. Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = 'plots/transformer_training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    log_f.write(f"Training curves saved to {plot_path}\n")

    log_f.write(f"\nTraining completed!\n")
    log_f.close()

    print(f"\nTraining completed! Results saved to:")
    print(f"  - Log file: {log_file}")
    print(f"  - Model: {model_save_path}")
    print(f"  - Plot: {plot_path}")

    return model

def generate_text(
        model_path='gpt_model.pt',
        train_path='train_shakespeare.txt',
        test_path='test_shakespeare.txt',
        prompt="the ",
        length=100,
        top_k=None,
        num_samples=3
):
    """
    Generate text using a trained GPT model.

    Args:
        model_path: Path to saved model checkpoint
        train_path: Path to training data (needed for DataHandler/vocab)
        test_path: Path to test data
        prompt: Starting text for generation
        length: Number of characters to generate
        top_k: If specified, use top-k sampling with this k value
        num_samples: Number of different sentences to generate
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract model parameters
    vocab_size = checkpoint['vocab_size']
    block_size = checkpoint['block_size']
    n_layer = checkpoint['n_layer']
    n_head = checkpoint['n_head']
    n_embd = checkpoint['n_embd']

    print(f"Loaded model from {model_path}")
    print(f"Parameters: vocab_size={vocab_size}, block_size={block_size}, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")

    # Initialize data handler
    data_handler = DataHandler(train_path, test_path, block_size)

    # Set device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    print(f"Using device: {device}")

    # Initialize model
    model = GPT(n_layer, n_head, n_embd, vocab_size, block_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Generate samples
    sampling_method = f"top-k (k={top_k})" if top_k else "regular sampling"
    print(f"\nGenerating {num_samples} samples using {sampling_method}")
    print(f"Prompt: '{prompt}'")
    print(f"Length: {length} characters\n")
    print("=" * 80)

    with torch.no_grad():
        for sample_idx in range(num_samples):
            generated_text = prompt

            for _ in range(length):
                # Encode the context (last block_size characters)
                context = generated_text[-block_size:]
                tokens = torch.tensor(data_handler.encoder(context))[None].to(device)

                # Get model predictions
                outputs = model(tokens)
                next_token_logits = outputs[0, -1, :]
                probabilities = torch.softmax(next_token_logits, dim=-1)

                # Sample next token
                if top_k:
                    # Top-k sampling
                    top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k)
                    top_k_probs = top_k_probs / torch.sum(top_k_probs)
                    sampled_idx = torch.multinomial(top_k_probs, num_samples=1).item()
                    next_token = top_k_indices[sampled_idx].item()
                else:
                    # Regular sampling
                    next_token = torch.multinomial(probabilities, num_samples=1).item()

                # Decode and append
                next_char = data_handler.decoder([next_token])
                generated_text += next_char

            print(f"Sample {sample_idx + 1}:")
            print(generated_text)
            print("-" * 80)

    return model, data_handler


if __name__=="__main__":
    torch.manual_seed(42)

    # Uncomment to train a new model
    train_model('train_shakespeare.txt', 'test_shakespeare.txt')

    # Generate text using the trained model
    print("\n" + "=" * 80)
    print("REGULAR SAMPLING")
    print("=" * 80)
    generate_text(
        model_path='gpt_model.pt',
        prompt="the ",
        length=100,
        num_samples=3
    )

    print("\n" + "=" * 80)
    print("TOP-K SAMPLING (k=5)")
    print("=" * 80)
    generate_text(
        model_path='gpt_model.pt',
        prompt="the ",
        length=100,
        top_k=5,
        num_samples=3
    )

