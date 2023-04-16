import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

text = "I'm trying to improve my footwork in boxing in order to have a better defense"
tokens = tokenizer.tokenize(text)
tokens = ["[CLS]"] + tokens + ["[SEP]"]
indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
segments_ids = [1] * len(tokens)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    
attentions = outputs.attentions[0];
layers = len(attentions[0][0])
heads = len(attentions[0])


attention = attentions[0,5, :].detach().numpy()  # get attention of first head and second layer
tokens = tokens[1:-1]  # remove special tokens

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(attention)

# add labels to x and y axes
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens)
ax.set_yticklabels(tokens)

# rotate x-axis labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

plt.show()
