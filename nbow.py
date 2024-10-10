import collections
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm

seed = 1231

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

max_length = 128

# the strings must be tokenized before they can be used in the model
def tokenize(input, tokenizer, max_length):
    tokens = tokenizer(input["text"])[:max_length]
    return {'tokens': tokens}

train_data = train_data.map(
    tokenize, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)

test_data = test_data.map(
    tokenize, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)

# split into train and valid, so that we can give the model a validation set that it hasnt seen yet and it doesnt overfit
test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

# builds the vocabulary, removing words that appear less than 5 times (to prevent having to many unique tokens)
# special token <unk> is used for unknown words (words that show up less than 5 times), and <pad> is used for padding
min_freq = 5
special_tokens = ["<unk>", "<pad>"]

vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)

unk_index = vocab["<unk>"]
pad_index = vocab["<pad>"]

vocab.set_default_index(unk_index)


# gives each token a unique id based on its index
def numericalize_input(input, vocab):
    ids = vocab.lookup_indices(input["tokens"])
    return {"ids": ids}

train_data = train_data.map(numericalize_input, fn_kwargs={"vocab": vocab})
valid_data = valid_data.map(numericalize_input, fn_kwargs={"vocab": vocab})
test_data = test_data.map(numericalize_input, fn_kwargs={"vocab": vocab})

# turns the data into torch tensors because pytorch doesnt operate on strings or integers, it operates on tensors
train_data = train_data.with_format(type = "torch", columns = ["ids", "label"])
valid_data = valid_data.with_format(type = "torch", columns = ["ids", "label"])
test_data = test_data.with_format(type = "torch", columns = ["ids", "label"])

# seperates data into batches and performs neccesary operations to make the data ready for the model
# pads the sequences so that they are all the same length
def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids" : batch_ids, "label" : batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fun = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        collate_fn = collate_fun,
        shuffle = shuffle
    )
    return data_loader


batch_size = 512

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

# building the models
# NBoW = Neural Bag of Words

class NBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc = nn.Linear(embedding_dim, output_dim)
    
    def forward(self, ids):
        # ids = [batch_size, seq_len]
        embedded = self.embedding(ids)
        # embedded = [batch_size, seq_len, embedding_dim]
        pooled = torch.mean(embedded, dim=1)
        # pooled = [batch_size, embedding_dim]
        prediction = self.fc(pooled)
        # prediction = [batch_size, num_classes]
        return prediction


# initializing the model with the desired parameters
vocab_size = len(vocab)
embedding_dim = 300
output_dim = len(train_data.unique("label"))

model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f'The model has {count_parameters(model):,} trainable parameters')

vectors = torchtext.vocab.GloVe()

pretrained_embeddings = vectors.get_vecs_by_tokens(vocab.get_itos())
# print(pretrained_embeddings.shape)
# print(model.embedding.weight)

model.embedding.weight.data = pretrained_embeddings

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# device is either cuda or the cpu depending on whether cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model = model.to(device)
criterion = criterion.to(device)

# training the model
def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="Training"):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

# testing the model
def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating"):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size 
    return accuracy

# running the model
n_epochs = 10
best_valid_loss = float("inf")
metrics = collections.defaultdict(list)

# Run the NBOW model

for epoch in range (n_epochs):
    train_loss, train_acc = train(train_data_loader, model, criterion, optimizer, device)
    valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "nbow.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")


# uncomment plt.show() to show the graphs

# graph the data based on losses
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(metrics["train_losses"], label="train loss")
# ax.plot(metrics["valid_losses"], label="valid loss")
# ax.set_xlabel("epoch")
# ax.set_ylabel("loss")
# ax.set_xticks(range(n_epochs))
# ax.legend()
# ax.grid()
# # plt.show()

# # graph the data based on accuracies
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(metrics["train_accs"], label="train accuracy")
# ax.plot(metrics["valid_accs"], label="valid accuracy")
# ax.set_xlabel("epoch")
# ax.set_ylabel("loss")
# ax.set_xticks(range(n_epochs))
# ax.legend()
# ax.grid()
# plt.show()

model.load_state_dict(torch.load("nbow.pt"))

test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)
print(f"NBOW: test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")


def predict_sentiment(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability


# Test the NBOW model on a sample text

text = "This movie is so bad"
print(predict_sentiment(text, model, tokenizer, vocab, device))

text = "This movie is terrific"
print(predict_sentiment(text, model, tokenizer, vocab, device))

text = "This film is not terrible, it's great!"

print(predict_sentiment(text, model, tokenizer, vocab, device))

text = "This film is not great, it's terrible!"

print(predict_sentiment(text, model, tokenizer, vocab, device))

