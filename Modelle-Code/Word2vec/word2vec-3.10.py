# Python 3.10

##### Quellen:
# https://radimrehurek.com/gensim/models/word2vec.html
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
# https://rare-technologies.com/word2vec-tutorial/
# https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.html
# https://radimrehurek.com/gensim/models/keyedvectors.html

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_with_legend.html#


import os
import re
from pprint import pprint
from gensim.models import Word2Vec, FastText

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


# Logging-Muster
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Preprocessing
# Modell braucht Liste von Listen
# sentences = [['first', 'sentence'], ['second', 'sentence']]



# Ordnerpfad zum Korpus eingeben
# ordner = 'korpus-word2vec-neu-saetze-mitAXSetc'   # mit Alexandertexten (84)
# ordner = 'korpus-word2vec-neu-saetze-ohneAXS'       # ohne Str Al (83)
# ordner = 'korpus-word2vec-neu-saetze-ohneAXSetc'  # ohne Alextexte (84-7 = 77)
# ordner = 'korpus-word2vec-neu-saetze-kleiner'       # 72

# ordner = 'korpus-word2vec-neu-saetze'  # mit AXS und Alextexten
# ordner = 'korpus-word2vec-nurAlex'
# ordner = 'korpus-w2v-neu-saetze-2'  # alles mit neuen stopwörtern


ordner = 'korpus-w2v-neu-saetze-alex-2'  # 7 alextexte neu


# Sätze in Liste speichern, Satzzeichen entfernen
saetze = []
for text in os.listdir(ordner):
    eine_datei = os.path.join(ordner, text)

    with open(eine_datei, 'r', encoding='utf-8') as liste:
        for satz in liste:
            satzzeichen_entfernen = re.sub(r'[.?!]', '', satz).split()
            saetze.append(satzzeichen_entfernen)


# Test
# print(saetze[:2])
print(len(saetze))



# Modell trainieren
# Parameterauswahl



model = Word2Vec(
    sentences=saetze,   # oder corpus_file
    vector_size=120,     # Dimensionalität der Wortvektoren
    window=5,           # Maximale Distanz zwischen Worten (current, predicted) in einem Satz
    min_count=5,        # Frequenz der Worte die ignoriert werden sollen
    workers=4,          # je mehr desto schneller wird trainiert
    sg=0,               # 1 skip-gram, 0 CBOW
    hs=0,               # hierarchisches Softmax 0 aus, 1 an
    epochs=10,
    seed=42             # Reproduzierbarkeit (100% aber nur mit workers=1 und PYTHONHASHSEED variable control)


    # alpha=   # learning rate
    # hashfxn=   # Hash function
)

## andere mögliche Parameter: min_alpha, seed, max_vocab_size, max_final_vocab, sample, trim_rule, sorted_vocab,
## batch_words, compute_loss, callbacks, shrink_windows

# Modell speichern
model.save('nk_model_mod')


# Modell laden
model = Word2Vec.load('nk_model_mod')  # um Modell weiter zu trainieren wenn nötig
# model.train()

# Häufigste Wörter
for i, w in enumerate(model.wv.index_to_key):
    if i == 10:
        break
    print(f"Wort #{i+1}/{len(model.wv.index_to_key)} ist {w}")  # / Größe Dictionary



# AUSWERTUNG
# default: cosine similarity

worte_aehnlich = model.wv.most_similar('rîtære', topn=50)  # Natur: wilde/wild oder lewe, tièr
                                                        # Kultur: zuht, hövischhèit, tugende, tugendlich
# pprint(worte_aehnlich)


# Worte Similarity für Wortfelder Annotation
simi = 0.50
for w, sim in worte_aehnlich:
    if sim > simi:
        print(f' Worte > {simi} {w, sim}')


wort_unpassend_1 = model.wv.most_similar(negative=['tièr'])
print(wort_unpassend_1)

# Unpassende Worte
wort_unpassend = model.wv.doesnt_match(['guot', 'rîtære', 'hant'])
pprint(wort_unpassend)

# Wortfeld ähnliche Worte
wortfeld = model.wv.most_similar(['hêrre', 'küning', 'mann'], topn=5)
pprint(wortfeld)

# Ähnlichkeit
aehnlichkeit_cosinus = model.wv.similarity(w1='vrouwe', w2='küninginne')
pprint(aehnlichkeit_cosinus)




# VISUALISIERUNG
# TSNE
# https://fortext.net/routinen/lerneinheiten/word2vec-mit-gensim
# https://fortext.net/routinen/methoden/word2vec-1
# https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d
# https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

# Plot ganzes Modell
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
# https://github.com/forTEXT/forTEXT.net/blob/main/word2vec/LerneinheitWord2Vec.ipynb
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html




# Plot ganzes Modell
def reduce_dimensions_TSNE(model):
    num_dimensions = 2  # Dimensionen (2 oder 3)

    # Labels und Vektoren
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)

    # TSNE
    tsne = TSNE(n_components=num_dimensions, random_state=42, perplexity=50)   #  init='pca',
                                                                            # random state für reproduzierbarkeit
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]

    # Plot ohne labels aber mit Farbverlauf
    plt.scatter(x_vals, y_vals, alpha=0.3, cmap='viridis', label='Wortverteilung')  # c=range(len(x_vals)),

    # plt.legend()
    plt.title('TSNE Visualisierung Modell')
    plt.colorbar()
    plt.tight_layout()
    plt.show()




def tsnescatterplot_wortfeld(model, wort, anzahl):
    # ähnliche Worte keys
    plot_worte = np.asarray(model.wv.most_similar(positive=([wort]), topn=anzahl))  # list[(str,float)] (key, similarity)

    # Wörter
    labels = [wort]  # eingabewort hinzufügen
    for w in plot_worte:
        labels.append(w[0])

    # Ähnliche Worte Vektoren anhand der Wörter
    w_vektoren = []
    for t in labels:
        wortvektor = model.wv.get_vector(t)
        w_vektoren.append(wortvektor)

    w_vektoren = np.asarray(w_vektoren)


    # TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=anzahl-1)  # random state für reproduzierbarkeit,
                                                                 # perplexity muss weniger als anzahl sein
    vectors = tsne.fit_transform(w_vektoren)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    #return x_vals, y_vals, labels

    # Scatterplot
    plt.scatter(x_vals, y_vals, alpha=0.5, color='skyblue')
    plt.scatter(x_vals[0], y_vals[0], color='red')  # Eingabewort rot

    for label, x, y in zip(labels, x_vals, y_vals):
        plt.annotate(label,  # label=label, s=scale
                     xy=(x, y),
                     xytext=(0, 0),  # Textplatzierung Punkt
                     textcoords='offset points',
                     ha='right',  # horizontale Ausrichtung Text
                     va='bottom')  # vertikale Ausrichtung Text

    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.title('TSNE Wortfeld')
    plt.tight_layout()
    plt.show()

# tsnescatterplot_wortfeld(model,'küning', 5)
# tsnescatterplot_wortfeld(model, 'rîtære', 30)
# tsnescatterplot_wortfeld(model, 'hërze', 20)



def tsnescatterplot_drei_wortfelder(model, wortliste, anzahl):  # bis zu drei Wortfelder
    # ähnliche Worte aus Wortliste
    aehnliche = []  # List[list[(str, float)]]
    for q in wortliste:
        plot_worte = model.wv.most_similar(positive=([q]), topn=anzahl)  # list[(str, float)]
        aehnliche.append(plot_worte)

    # Wörter
    labels = wortliste.copy()  # Eingabeworte in Liste kopieren für Plot
    for c in aehnliche:
        for word in c:
            labels.append(word[0])

    labels_np = np.asarray(labels)
    print(labels_np)

    # Vektoren
    w_veks = []
    for m in labels_np:
        veks = model.wv.get_vector(m)
        w_veks.append(veks)

    w_veks_np = np.asarray(w_veks)


    # TSNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=anzahl-1)  # random state für reproduzierbarkeit, perplexity muss weniger als anzahl sein
    vectors = tsne.fit_transform(w_veks_np)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]

    # Scatterplot

    plt.scatter(x_vals[:len(wortliste)], y_vals[:len(wortliste)], color='red', label='Eingabeworte')  # Eingabeworte
    plt.scatter(x_vals[(len(wortliste)):(anzahl+len(wortliste))], y_vals[(len(wortliste)):(anzahl+(len(wortliste)))],
                color='orange', alpha=0.5, label=f'Ähnlichste Worte {wortliste[0]}')  # alpha = transparenz
    plt.scatter(x_vals[(anzahl+len(wortliste)):((2*anzahl)+(len(wortliste)))],
                y_vals[(anzahl+len(wortliste)):((2*anzahl)+(len(wortliste)))],
                color='aqua', alpha=0.5, label=f'Ähnlichste Worte {wortliste[1]}')
    plt.scatter(x_vals[((2*anzahl)+len(wortliste)):((3*anzahl)+(len(wortliste)))],
                y_vals[((2*anzahl)+len(wortliste)):((3*anzahl)+(len(wortliste)))],
                color='green', alpha=0.5, label=f'Ähnlichste Worte {wortliste[2]}')


    for label, x, y in zip(labels, x_vals, y_vals,):
        plt.annotate(label,  # label=label, s=scale
                    xy=(x, y),
                    xytext=(5, 5),  # Textplatzierung Punkt
                    textcoords='offset points',
                    ha='right',  # horizontale Ausrichtung Text
                    va='bottom')  # vertikale Ausrichtung Text


    plt.legend()
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')
    plt.title('TSNE Wortfeld')
    plt.tight_layout()
    plt.show()

# Test
# tsnescatterplot_drei_wortfelder(model, ['hërze', 'lèid', 'dëgenhèit'], 30)



# Modell Fasttext für "natur" und "kultur"
# keine guten Ergebnisse


model = FastText(
    sentences=saetze,   # oder corpus_file
    vector_size=120,     # Dimensionalität der Wortvektoren https://rare-technologies.com/word2vec-tutorial/
    window=5,           # Maximale Distanz zwischen Worten (current, predicted) in einem Satz
    min_count=5,        # Frequenz der Worte die ignoriert werden sollen
    workers=4,          # je mehr desto schneller wird trainiert # = 1 für konsistenz (dauert aber sehr lang)
    sg=0,               # 1 skip-gram, 0 CBOW
    hs=0,               # hierarchisches Softmax 0 aus, 1 an
    epochs=10,
    seed=42             # Reproduzierbarkeit (100% aber nur mit workers=1 und PYTHONHASHSEED variable control)
    )

model.save('nk_model_ft')

model = FastText.load('nk_model_ft')

worte_aehnlich_ft_n = model.wv.most_similar('natûre', topn=30)
worte_aehnlich_ft_k = model.wv.most_similar('kultûre', topn=30)
print(worte_aehnlich_ft_n)
print(worte_aehnlich_ft_k)





