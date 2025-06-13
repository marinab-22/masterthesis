# Python 3.8 weil Gensim 3.8.3

### Quellen:
# https://radimrehurek.com/gensim/models/ldamodel.html
# https://radimrehurek.com/gensim_3.8.3/models/wrappers/ldamallet.html
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html
# https://radimrehurek.com/gensim/models/coherencemodel.html
# https://radimrehurek.com/gensim_3.8.3/models/wrappers/ldamallet.html

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html
# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# https://matplotlib.org/stable/users/explain/colors/colormaps.html




import os
from pprint import pprint

# Gensim
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# Mallet
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet

# Vis
import numpy as np
import matplotlib.pyplot as plt

# import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Daten vorbereiten
# Jedes doc muss Liste in Liste von Docs sein
# corpus = [[d1],[d2],[d3]]

# ordner = 'korpus-topicM-neu-mitAXSetc'  # ganzes Korpus mit allen Alextexten (84)
# ordner = 'korpus-topicM-neu-ohneAXS'  # Korpus ohne Straßburger Alexander (-100 unique tokens) !  # 4 Topics umass -0.03
# ordner = 'korpus-topicM-neu-ohneAXSetc'  # Korpus ohne Alexandertexte (84 -7 Texte)  # 4 Topics für u mass -0.09

# ordner = 'korpus-topic-nurStrAl'
# ordner = 'korpus-topic-nurAlex'  # 7 Alexandertexte

# neueste:
# ordner = 'korpus-tm-7Alex'  # 7
# ordner = 'korpus-topicM-neu-1'  # keine guten Ergebnisse
ordner = 'korpus-topic-tm-2'
# ordner = 'korpus-topic-tm-3'

docs = []

for t in os.listdir(ordner):
    zsm = os.path.join(ordner, t)
    with open(zsm, 'r', encoding='utf-8') as eins:
        k = eins.read().split()
        docs.append(k)

# Test
# print(len(docs))
# print(docs[0][:20])  # erstes doc


# Dictionary erstellen
dictionary = Dictionary(docs)

# Worte filtern
# dictionary.filter_extremes(no_below=20, no_above=0.5)

# Bag-of-words
corpus = [dictionary.doc2bow(doc) for doc in docs]

ut = len(dictionary)
num_docs = len(corpus)

print(f'Anzahl unique tokens: {ut}')
print(f'Anzahl Dokumente: {num_docs}')

# Dictionary
temp = dictionary[0]  # Dict laden
id2word = dictionary.id2token  # für u_mass coherence nötig



##### KONFIGURATIONEN

## Korpus 7 Alexandertexte:
    ## topics = 4, chunksize = 1, passes = 15
    ## Ergebnis: Topic mit "ross" + "freislich" --> gut, aber nicht ausreichend



# Anzahl der Topics für gensim LDA und Mallet
num_topics = 4  # beste Ergebnisse

model = LdaModel(
    corpus=corpus,
    id2word=id2word,  # id2word=dictionary oder id2word
    chunksize=1,  # Dokumente pro Trainingschunk
    alpha='auto',  # symmetric (default) asymmetric, auto
    eta='auto',  # symmetric, auto
    num_topics=num_topics,
    passes=15,
    random_state=42  # Reproduzierbarkeit
)

print_topics = model.print_topics(num_words=40)
pprint(print_topics)

top_topics = model.top_topics(corpus, coherence='u_mass', topn=10)
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print(f'Durchschnittliche Topic Kohärenz u_mass LDA: {avg_topic_coherence}')

# CoherenceModel für die Evaluation
# U_mass empfohlen: https://www.baeldung.com/cs/topic-modeling-coherence-score
cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
coherence = cm.get_coherence()  # get coherence value
print(f'Kohärenzwert u_mass LDA: {coherence}')

# model = model.save('tm-topics2-rs50')
# model = model.load('tm-topics2-rs50')



# Mallet

mallet_path = 'C:/mallet-2.0.8/bin/mallet'
model_mallet = LdaMallet(
    mallet_path,
    corpus=corpus,
    num_topics=num_topics,
    id2word=id2word,  # dictionary oder id2word
    iterations=15,
    random_seed=42,  # Konsistenz/reproduzierbarkeit
)

# Topics anzeigen
print_topics_mallet = model_mallet.print_topics(num_words=30)
pprint(print_topics_mallet)

# .top_topics geht bei Mallet nicht
# top_topics_mallet = model_mallet.top_topics(corpus, topn=10)
# avg_topic_coherence_mallet = sum([t[1] for t in top_topics_mallet]) / num_topics
# print(f'Durchschnittliche Topic Kohärenz u_mass Mallet: {avg_topic_coherence_mallet}')

# Kohärenzwert

cm = CoherenceModel(model=model_mallet, corpus=corpus, coherence='u_mass')
coherence = cm.get_coherence()  # get coherence value
print(f'Kohärenzwert u_mass Mallet: {coherence}')




##### Visualisierung
# pyLDAvis braucht mehrere parameter, müsste docs als liste von strings speichern um doc_lengths zu kriegen
# Document-Topic-Distribution
# in welchem doc sind welche topics wie verteilt


# Wortanteil/Wortgewichtung pro Topic

# Wörter und Gewichtungen für den Plot vorbereiten

# je nachdem model oder model_mallet einfügen
# LDA gensim
topics = model.show_topics(formatted=False, num_words=40, num_topics=num_topics)  # formatted = False: List[Tuple(w, prob)]
# Mallet
# topics = model_mallet.show_topics(formatted=False, num_words=30, num_topics=num_topics)                                                                                    # formatted = True: string

# Topic-Wörter und -Gewichtungen
topic_worte = []
topic_gew = []

for topic in topics:
    topic_worte.append(topic[1])  # topics Tuple(str, float)
                                  # topic [0] wäre Topicnummer plus wort-gew-paare
    wort_gew = []
    for wort in topic[1]:
        wort_gew.append(wort[1])  # gewichtung

    topic_gew.append(np.array(wort_gew))  # Umwandeln in numpy array

# print(topics)       # (0, [('sprëchen', 0.045156498),
# print(topic_worte)  # ('sprëchen', 0.045156498),
# print(topic_gew)    # [array([0.0451565 , 0.02615629,


# Gewichtung der Worte pro Topic
# Plotten der Topicworte und Anteile als gestapelte Balken
fig, ax = plt.subplots(figsize=(10, 6))

# Stacked bar plot
bottom = np.zeros_like(topic_gew[0])
for i, weights in enumerate(topic_gew):
    ax.barh(range(len(weights)), weights, align='center', label=f'Topic {i+1}', left=bottom)
    bottom += weights

ax.set_yticks(np.arange(len(topic_worte[0])))
ax.set_yticklabels([w[0] for w in topic_worte[0]])
ax.invert_yaxis()
ax.set_xlabel('Topic-Anteile', fontweight='bold')
ax.set_ylabel('Wörter', fontweight='bold')
ax.set_title('Wortverteilung pro Topic')
ax.legend()
plt.show()




# Topicverteilung pro Dokument
# Dokument-Topic-Verteilung
mallet_doc_topics_datei = model_mallet.fdoctopics()  # Mallet

doc_topic_proportions = []  # jedes doc muss liste[(topicID, prob)] sein in einer liste
for doc in corpus:
    #  LDA
    doc_topics = model.get_document_topics(doc)  # List[(topicID, probability)]
    doc_topic_proportions.append(doc_topics)

pprint(doc_topic_proportions)



# Werte für Plot

# Umwandeln in 2-dimensionales numpy array
data = np.zeros((num_docs, num_topics))
for i, doc in enumerate(doc_topic_proportions):  # i = index des docs, doc = Liste von Tupeln
    for topic, prob in doc:  # Schleife über jedes Tupel
        data[i, topic] = prob  # prob und topic für i (doc) werden im array gespeichert

# gestapelter barplot
plt.figure(figsize=(18, 6))

bottom = np.zeros(num_docs)  # neuer array mit Nullen gefüllt für Unterseite der Balken im Diagramm
for i in range(num_topics):  # Schleife über jedes Topic
    plt.bar(range(num_docs), data[:, i], bottom=bottom, label=f'Topic {i + 1}')  # data[:,i]: Höhe der Balken für Topic i
    bottom += data[:, i]  # gestapelte Balken: bottom ist ende des ersten und anfang des zweiten balkens zb

plt.xlabel('Dokumente', fontweight='bold')
plt.ylabel('Topic-Anteile', fontweight='bold')
plt.title('Dokument-Topic-Verteilung', fontsize=14)
plt.xticks(range(0, num_docs, 4), [i+1 for i in range(0, num_docs, 4)])
plt.legend(title='Topics', loc='upper right')
plt.tight_layout()
plt.show()


# Heatmap als Alternative zum gestapelten Balkendiagramm
# Erstellen der Heatmap (braucht auch umwandlung in 2d array)

plt.figure(figsize=(8, 8))
plt.imshow(data, cmap='Blues', aspect='auto')
plt.yticks(range(0, num_docs, 4), [i+1 for i in range(0, num_docs, 4)], fontsize=10)
plt.xticks(range(num_topics), [i+1 for i in range(num_topics)], fontsize=10)
plt.colorbar(label='Topic-Anteile')
plt.xlabel('Topics', fontweight='bold')
plt.ylabel('Dokumente', fontweight='bold')
plt.title('Dokument-Topic-Verteilung', fontsize=14)
plt.tight_layout()
plt.show()






##### Zusätzliche Infos

# umass vs cv coherence
# https://www.baeldung.com/cs/topic-modeling-coherence-score
# https://github.com/dice-group/Palmetto/issues/13#issuecomment-371553052
# --> Probleme cv (aber nur in Palmetto 2018 bzw Lösung 2023)
# --> gensim 3.8.3 release mai 2020 --> Fehler besteht evtl auch hier noch
# umass nur zum modell/paramtervergleich
# intrinsische messung https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/





