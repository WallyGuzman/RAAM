from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def find_nn(decoded, gold_sentences, word_dict, word_dim_size):
    #print(gold_sentences)
    word_vectors = np.zeros((len(word_dict.keys()), word_dim_size), dtype=float)

    for iter, key in enumerate(list(word_dict.keys())):
        temp = np.copy(word_dict[key])
        np.expand_dims(temp, axis=0)
        word_vectors[iter][:] = temp

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(word_vectors, list(word_dict.keys()))

    for iter, sentence_vecs in enumerate(decoded[:]):
        pred = classifier.predict(sentence_vecs)
        #print(pred.shape)
        print("Correct:")
        print(gold_sentences[iter])
        print("Prediction:")
        print(pred)
        print(iter)