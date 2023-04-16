import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from sklearn.manifold import TSNE


# Load pre-trained Word2Vec model
model = api.load('word2vec-google-news-300')

# Select some words to plot
words = ['music', 'literature', 'philosophy', 'sociology', 'art', 'linguistics',  'storm', 'earthquake', 'rain', 'stock', 'bonds', 'swaption', 'trader', 'growth']


# Get the vectors for the selected words
word_vectors = np.array([model[w] for w in words])


# Use PCA to reduce the word embeddings to 2 dimensions
tsne = TSNE(n_components=2,perplexity=5)
word_vectors_2d = tsne.fit_transform(word_vectors)



#plotting
plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(word_vectors_2d[i, 0], word_vectors_2d[i, 1]))
plt.show()
print("finished!!!")
