import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from ptncn_model import PTNCN  # Assuming PTNCN is already translated to PyTorch
import torchtext

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_ptb_data():
    data = load_dataset('ptb_text_only', 'penn_treebank')
    data_train = data['train']
    # data_valid = data['validation']

    max_sentence_length = 0

    def yield_ptb_tokens():
        nonlocal max_sentence_length
        for example in data_train:
            if len(example['sentence']) > max_sentence_length:
                max_sentence_length = len(example['sentence'])
            yield example['sentence']

    vocab = torchtext.vocab.build_vocab_from_iterator(yield_ptb_tokens())

    # preprocess data_train into a tensor (N, L) of vocab indices, where L is the max length of a sentence
    jagged_train_list = [torch.tensor([vocab[token] for token in example['sentence']]) 
                  for example in data_train]


    padded_data_train = torch.vstack([F.pad(t, (0, max_sentence_length - t.shape[0]), mode="constant", value=-1)
                                      for t in jagged_train_list])

    return vocab, padded_data_train


# one-hot encode a tensor of indices. any negative values get mapped to a vector of all zeros
def one_hot_padding(tensor, length):
    mask = (tensor >= 0).type(torch.float32)
    tensor2 = torch.where(tensor >= 0, tensor, 0)
    tensor3 = F.one_hot(tensor2, length)
    return tensor3 * mask.unsqueeze(-1)



def train_ptncn():
    # Training hyperparameters
    num_epochs = 10
    batch_size = 50

    # Model hyperparameters
    lambda_ = 0.001
    beta = 0.15
    gamma = 0.01

    # Optimizer hyperparameters
    lr = 0.075
    momentum = 0.95
    use_nesterov = True

    vocab, data_train = get_ptb_data()
    vocab_size = len(vocab)

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)


    # Model initialization
    model = PTNCN(input_size=vocab_size, hidden_size=256, lambda_=lambda_, beta=beta, gamma=gamma)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=use_nesterov)

    print('--------------------')
    # print(f'Train: {len(data_train)}')
    # print(f'Valid: {len(data_valid)}')


    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for X in train_loader:
            for t in range(X.shape[1]):
                optimizer.zero_grad()

                X_t = X[:, t]
                mask_t = (X_t >= 0).type(torch.float32)
                X_enc_t = one_hot_padding(X_t, vocab_size)
                model.forward(X_enc_t, mask_t)
                model.compute_correction()

                optimizer.step()
        break


if __name__ == '__main__':
    set_random_seed(628318)
    train_ptncn()