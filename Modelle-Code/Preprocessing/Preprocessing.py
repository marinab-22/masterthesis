
### Preprocessing
# viertespalte extrahieren (precomputed lemmas laut RNNtagger)
# großgeschriebene Worte (Namen + Orte) entfernen
# doppelungen aus stopwortliste großgeschrieben entfernen



# lemma liste ordner: viertespalte extrahieren
import os
import nltk
from nltk.tokenize import word_tokenize
import re
from nltk.tokenize import sent_tokenize

# nltk.download('punkt')


# Vierte Spalte extrahieren (precomputed Lemmas aus dem Tagger)
def viertespalte(in_ordner, out_ordner):
    # Output Ordner erstellen
    if not os.path.exists(out_ordner):
        os.makedirs(out_ordner)

    for datei in os.listdir(in_ordner):
        korpus_ordner_tagged = os.path.join(in_ordner, datei)
        neuer_ordner = os.path.join(out_ordner, datei)

        with open(korpus_ordner_tagged, 'r', encoding='utf-8') as kt:
            zeilen = kt.readlines()

        with open(neuer_ordner, 'w', encoding='utf-8') as neue_dateien:
            for l in zeilen:
                spalten = l.split('\t')
                if len(spalten) >= 4:
                    neue_dateien.write(spalten[3].strip() + '\n')

# Korpus
viertespalte('korpus-tagged-ganz', 'korpus-tagged-ganz-viertespalte')

# Stopwortliste
# (Input Ordner erstellen + Ergebnis aus dem Output Ordner holen)
# viertespalte('gmh-stopwords-extended-tagged', 'stopwords-extended-viertespalte')

# output file umbenannt und in Projektordner gespeichert: gmh-stopwords-extended-tagged-viertespalte

# txt-neu-out (Eigene Satzzeichen gesetzt bei NBB NLA DIO TRY)
# viertespalte('txt-neu-out', 'txt-neu-spalte')



# großgeschriebene worte (namen) extrahieren

def worte_gross(in_ordner, out_datei):
    # Regex für großgeschriebene Worte
    pattern = re.compile(r'\b[A-Z]\w*\b')

    with open(out_datei, 'w', encoding='utf-8') as neue_liste:
        for datei in os.listdir(in_ordner):
            dateien = os.path.join(in_ordner, datei)
            with open(dateien, 'r', encoding='utf-8') as d:
                text = d.read()
                worte_extrahieren = re.findall(pattern, text)  # Großgeschriebene Worte finden (Namen, Orte)
                for wort in worte_extrahieren:
                    w = wort.lower()
                    neue_liste.write(w + '\n')


worte_gross('korpus-tagged-ganz-viertespalte', 'stopwords-caps')
# worte_gross('korpus-tagged-ganz-viertespalte', 'TEST-stopwords-caps')


# bei getagged stoppwörtern und bei großgeschriebenen Namen: stopwords-caps + gmh-stopwords-extended-tagged-viertespalte
def doppelungen_entfernen(in_datei, out_datei):
    with open(in_datei, 'r', encoding='utf-8') as i:
        words = i.read().split()

    worte_einzeln = set(words)  # Set verhindert Doppelungen, unsortiert

    with open(out_datei, 'w', encoding='utf-8') as neue_datei:
        for w in worte_einzeln:
            neue_datei.write(w + '\n')

# Stopwörter
# doppelungen_entfernen('stopwords-caps', 'stopwords-caps-bereinigt')
# doppelungen_entfernen('stopwords-caps', 'stopwords-caps-bereinigt-neu')

# Stopwörter
# doppelungen_entfernen('gmh-stopwords-extended-tagged-viertespalte.txt', 'gmh-stopwords-extended-tagged-viertespalte-cleaned')
# doppelungen_entfernen('gmh-stopwords-extended-tagged-viertespalte.txt', 'gmh-stopwords-extended-tagged-viertespalte-cleaned-neu')


# Stopwortlisten joinen


# TopicM:
# stopwortliste mit . ? !
# Word2vec:
# stopwortliste ohne . ? !



def stopwords_entfernen(in_ordner, stopwords_liste, out_ordner):  # dauert sehr lang
    # Output Ordner estellen
    if not os.path.exists(out_ordner):
        os.makedirs(out_ordner)

    # Stopwortliste aus der Datei lesen
    with open(stopwords_liste, 'r', encoding='utf-8') as sw:
        stopwords = sw.read().split()

    # Dateien im in_ordner
    for datei in os.listdir(in_ordner):
        korpus_ordner = os.path.join(in_ordner, datei)
        neuer_ordner = os.path.join(out_ordner, datei)
        # Tokenisieren + Stopwörter entfernen
        result = []
        with open(korpus_ordner, 'r', encoding='utf-8') as dateien:
            for line in dateien:
                words = line.split()
                words = word_tokenize(line)
                filtered_words = [wort for wort in words if wort.lower() not in stopwords]
                result.extend(filtered_words)  # nimmt mehrere Objekte raus (vs .append() nimmt nur eines als ganzes)

        with open(neuer_ordner, 'w', encoding='utf-8') as neue_dateien:
            neue_dateien.write(' '.join(result))  # Fließtext


# remove_stopwords('korpus', 'gmh_stopwords_extended.txt', 'stopwords-removed')
# remove_stopwords('korpus-tagged-lemmas', 'gmh-stopwords-lemmas-liste.txt', 'korpus-stopwords-removed-tagged')
# remove_stopwords('korpus', 'stopwords-join-word2vec.txt', 'korpus-word2vec')
# remove_stopwords('korpus-tagged-ganz-viertespalte', 'stopwords-word2vec.txt', 'korpus-tagged-ganz-viertespalte-word2vec-ohnesw')
# remove_stopwords('korpus-tagged-ganz-viertespalte-word2vec-ohnesw', 'stopwords-groß-bereinigt.txt', 'korpus-tagged-ganz-viertespalte-word2vec-ohnesw-ohnenamen')


# TopicM
# stopwords_entfernen('korpus-tagged-ganz-viertespalte', 'stopwords-join-topicM-neu.txt', 'korpus-topicM-neu')
# stopwords_entfernen('korpus-tagged-ganz-viertespalte', 'stopwords-join-topicM-neu.txt', 'korpus-topicM-neu-1')
stopwords_entfernen('korpus-tagged-ganz-viertespalte', 'stopwords-join-topicM-neu.txt', 'korpus-topicM-neu-2')  # ohne vone, al, mit sprechen
# stopwords_entfernen('korpus-tagged-ganz-viertespalte', 'stopwords-gut-tm.txt', 'korpus-topicM-neu-3')  # ohne vone, al, sprechen

# Word2vec
# stopwords_entfernen('korpus-tagged-ganz-viertespalte', 'stopwords-join-word2vec-neu.txt', 'korpus-word2vec-neu')
stopwords_entfernen('korpus-tagged-ganz-viertespalte', 'stopwords-w2v-neu-2.txt', 'korpus-w2v-neu-2')


# txt-neu-out
# stopwords_entfernen('txt-neu-spalte', 'stopwords-join-word2vec-neu.txt', 'txt-neu-w2v')
# stopwords_entfernen('txt-neu-spalte', 'stopwords-join-topicM-neu.txt', 'txt-neu-tm')
# stopwords_entfernen('txt-neu-spalte', 'stopwords-join-topicM-neu.txt', 'txt-neu-tm-1')
stopwords_entfernen('txt-neu-spalte', 'stopwords-join-topicM-neu.txt', 'txt-neu-tm-2')  # ohne vone, al, mit sprechen
# stopwords_entfernen('txt-neu-spalte', 'stopwords-gut-tm.txt', 'txt-neu-tm-3')  # ohne vone, al, sprechen

stopwords_entfernen('txt-neu-spalte', 'stopwords-w2v-neu-2.txt', 'txt-w2v-neu-2')


# Word2vec Sätze
def saetze_tokens(in_ordner, out_ordner):
    # Output Ordner erstellen
    if not os.path.exists(out_ordner):
        os.makedirs(out_ordner)

    for datei in os.listdir(in_ordner):
            wv_ordner = os.path.join(in_ordner, datei)
            neuer_ordner = os.path.join(out_ordner, datei)

            with open(wv_ordner, 'r', encoding='utf-8') as dateien:
                text = dateien.read().strip()
                saetze_tokenized = sent_tokenize(text)  # nltk tokenizer für Sätze

            # In neuen Ordner schreiben
            with open(neuer_ordner, 'w', encoding='utf-8') as neue_dateien:
                for satz in saetze_tokenized:
                    neue_dateien.write(satz + '\n')





# saetze_tokens('korpus-tagged-ganz-viertespalte-word2vec-ohnesw', 'korpus-tagged-ganz-viertespalte-word2vec-ohnesw-saetze')


# Word2vec
# saetze_tokens('korpus-word2vec-neu', 'korpus-word2vec-neu-saetze'
saetze_tokens('korpus-w2v-neu-2', 'korpus-w2v-neu-saetze-2')

# txt-neu-out
# saetze_tokens('txt-neu-w2v', 'txt-neu-w2v-saetze')
saetze_tokens('txt-w2v-neu-2', 'txt-w2v-neu-saetze-2') # zum Korpus hinzugefügt