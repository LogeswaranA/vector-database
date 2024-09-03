import numpy as np
from vector_store import VectorStore

vector_store = VectorStore()

sentences = [
    "hey I love apple",
    "apple is a fruit",
    "mango is also a fruit",
    "comman man loves both mango and fruit"
]

vocabulary = set()
for sentence in sentences:
    words = sentence.lower().split()
    vocabulary.update(words)

print(vocabulary)

sorted_vocabulary = sorted(vocabulary)

#Now let's convert this words into words with index

words_to_index = { word : i for i, word in enumerate(sorted_vocabulary)}

print(words_to_index)

sentence_vectors = {}
for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[words_to_index[token]] +=1
    sentence_vectors[sentence] = vector

print("sentence_vectors:::",sentence_vectors)


#Create a vector store with sentence to vector mapping
for sentence,vector in sentence_vectors.items():
    vector_store.add_vector(sentence,vector)

print("vector_store::::",vector_store)

#Create a similarity search for new query

query_sentence = "apple is my"
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()

for token in query_tokens:
    if token in words_to_index:
        query_vector[words_to_index[token]] += 1


similar_sentence_search = vector_store.find_similar_vectors(query_vector,num_results=2)
print(similar_sentence_search)


