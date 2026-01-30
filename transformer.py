import math
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from dataset import DataHandler


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
        epochs=10
):            
                    
    
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
    print('Using device:', device)


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
    test_accuracies = []
    test_losses = []
    for ep in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):            
            #### YOUR CODE HERE ####
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            accumalated_test_loss = 0.0
            number_of_batches = 0
            for i, batch in enumerate(tqdm(test_loader)):
                #calculate test accuracy
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs) #TODO check later if squeeze is needed
                highest_probability_token = torch.argmax(outputs, dim=-1)
                correct += (highest_probability_token == targets).float().mean()
                loss = criterion (outputs.view(-1, vocab_size), targets.view(-1)) 
                accumalated_test_loss += loss.item()
                number_of_batches += 1
                #print test loss
            test_accuracy = correct / (i + 1)
            print(f"Epoch {ep+1}/{epochs}, Test Loss: {accumalated_test_loss / number_of_batches}, Test Accuracy: {test_accuracy}")
            test_accuracies.append(test_accuracy)
            test_losses.append(accumalated_test_loss / number_of_batches)
            # Complete the sentence:
            sentence="the "
            for i in range(3):
                new_sentence = sentence
                for i in range(20):                            
                        tokens = torch.tensor(data_handler.encoder(sentence[-block_size:]))[None]
                        #### YOUR CODE GOES HERE ####
                        tokens = tokens.to(device)
                        outputs = model(tokens)
                        next_token_logits = outputs[0, -1, :]
                        probabilities = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probabilities, num_samples=1).item()
                        next_char = data_handler.decoder([next_token])
                        new_sentence += next_char
                print('new_sentence: ', new_sentence)


            # Comple the sentence only considering the top k characters when sampling:
            for i in range(3):
                for i in range(20):
                    tokens = torch.tensor(data_handler.encoder(sentence[-block_size:]))[None]
                    tokens = tokens.to(device)
                    outputs = model(tokens)
                    next_token_logits = outputs[0, -1, :]
                    top_5_probabilities, _ = torch.topk(torch.softmax(next_token_logits, dim=-1), k=5)
                    top_5_probabilities = top_5_probabilities / torch.sum(top_5_probabilities)
                    next_token = torch.multinomial(top_5_probabilities, num_samples=1).item()
                    next_char = data_handler.decoder([next_token])
                    new_sentence += next_char
        # plot the test and reaining accuracies and losses



if __name__=="__main__":
    torch.seed(42)
    train_model('train.txt', 'test.txt')
    

