import pandas as pd
import os
import re
import string
import csv
import logging as log
import matplotlib.pyplot as plt
import nltk
from datetime import datetime
from numpy import asarray, zeros, nan
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


DATA_DIR = f"{os.path.dirname(os.path.abspath(__file__))}\\..\\..\\scraper\\data\\"
HIST_FL_PATH = f"{os.path.dirname(os.path.abspath(__file__))}\\..\\historical_2016.csv"
GLV_FL_PATH = f"{os.path.dirname(os.path.abspath(__file__))}\\..\\glove_s100.txt"
GRAPH_PATH = f"{os.path.dirname(os.path.abspath(__file__))}\\..\\graphs\\"
MAX_LEN = 100
POSITIVE_RELEVANCE = 2
NEGATIVE_RELEVANCE = 0
IRRELEVANT = 1
VAR_MIN = 0.1
STOP_WORDS = ['de', 'a', 'o', 'que', 'e', 'é', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houverá', 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem', 'temos', 'tém', 'tinha', 'tínhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos', 'teriam', 'petrobras', 'reuters', 'pontos', 'ibovespa', 'após', 'paulo', 'nesta', 'mercado', 'ações', 'on', 'índice', 'petróleo', 'pn', 'vale', 'financeiro', 'reais', 'bilhões', 'destaques', 'sessão', 'brasil', 'preços', 'sobre', 'bolsa', 'cento', 'itaú', 'china', 'bradesco', 'ainda', 'unibanco', 'enquanto']
# STOP_WORDS = ['de', 'a', 'o', 'que', 'e', 'é', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'está', 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houverá', 'houveremos', 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem', 'temos', 'tém', 'tinha', 'tínhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos', 'teriam']

def main(var):
    log.info("Starting Process")
    corpus = gather_data(var)
    # bag_of_words = extract_bag_of_words(corpus["Conteudo"])
    # new_stop_words = find_new_stop_words(bag_of_words)
    # stop_words = nltk.corpus.stopwords.words("portuguese")
    # for word in new_stop_words:
    #     stop_words.append(word)
    text = corpus["Conteudo"].apply(lambda x: treat_content(x, STOP_WORDS))
    # Set the output sentiment to be a 3 columns df
    sentiment = to_categorical(corpus["Sentimento"])
    text_train, text_test, sentiment_train, sentiment_test = train_test_split(text, sentiment, test_size=0.10)
    text_train, text_test, embedding_layer = prepare_embedding_layer(text_train, text_test)
    log.info("Test and Train sets ready")

    # Monta o Modelo LSTM
    model = Sequential()
    log.info("\nBuilding Model")
    model.add(embedding_layer)
    log.info("\nEmbedding Layer ready")
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    log.info("\nLSTM Layer ready")
    model.add(Dense(3,  activation="softmax"))
    log.info("\nDense Layer ready")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    log.info("\nCompilation Completed")
    print(model.summary())

    # Treina o Modelo e realiza o Teste de Avaliação
    log.info("\nStart training")
    history = model.fit(text_train, sentiment_train, batch_size=128, epochs=20, verbose=1, validation_split=0.1)
    log.info("\nModel Trained!")
    score = model.evaluate(text_test, sentiment_test)
    log.info(f"\nLoss: {score[0]}")
    log.info(f"\nAccuracy: {score[1]}")
    log.info("\nPlotting history")
    result_plots(history, var)
    log.info("\nPlotting completed!")
    model.save(f"{os.path.dirname(os.path.abspath(__file__))}\\..\\SentimentModel.h5")


def gather_data(var):
    log.info("------------------------------")
    log.info("Gathering data for analysis")
    log.info("------------------------------")

    # Read each news file and build a df containing the "Title", "Date" and "Text"
    log.info("Reading every news file...")
    news_list = []
    for filename in os.listdir(DATA_DIR):
        with open(DATA_DIR+filename) as csv_file:
            read_csv = csv.reader(csv_file, delimiter='¨')
            for row in read_csv:
                news_list.append(
                    {
                        "Titulo": row[0],
                        "Data": row[1],
                        "Identificação": row[2],
                        "Conteudo": row[3],
                        "Sentimento": nan
                    }
                )
    news_data = pd.DataFrame(news_list, columns=["Titulo", "Data", "Conteudo", "Sentimento"])

    # Prepare the historical data scrapped from the site
    log.info("Reading the historical values...")
    hist_data = pd.read_csv(HIST_FL_PATH)
    hist_data["Data"] = hist_data["Data"].apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%Y-%m-%d'))
    hist_data["Último"] = hist_data["Último"].apply(lambda x: float(x.replace(",", ".")))
    hist_data["Var%"] = hist_data["Var%"].apply(lambda x: float(x.replace(",", ".").split("%")[0]))

    # Merge the historical values with the news, excluding days without news
    partial_corpus = news_data.merge(right=hist_data, how="left", on="Data")
    partial_corpus = partial_corpus.sort_values(by="Data").reset_index(drop=True)
    partial_corpus["Var%"] = partial_corpus["Var%"].apply(lambda x: x if not pd.isna(x) else 0)

    # Update the "Sentimento" column based on the values of the "Var%" column
    log.info("Updating the 'Sentimento' column...")
    partial_corpus.update(sentiment_logic(partial_corpus, var))
    corpus = partial_corpus.copy()
    log.info("Data gathered!")
    return corpus


def sentiment_logic(corpus_df, var):
    sentiment_list = []
    for index, row in corpus_df.iterrows():
        # The last day is irrelevant to the training, as it lacks info
        if index == len(corpus_df)-1:
            sentiment_list.append(IRRELEVANT)

        # If there was no Variance in the stoke = irrelevant
        elif corpus_df.loc[index]["Var%"] == 0:
            sentiment_list.append(IRRELEVANT)

        # If the stoke was descending before the news and it started ascending after the news = positive
        elif corpus_df.loc[index]["Var%"] < 0 and corpus_df.loc[index+1]["Var%"] > 0:
            sentiment_list.append(POSITIVE_RELEVANCE)

        # If the stoke was ascending before the news and it started descending after the news = negative
        elif corpus_df.loc[index]["Var%"] > 0 and corpus_df.loc[index+1]["Var%"] < 0:
            sentiment_list.append(NEGATIVE_RELEVANCE)

        # if the stoke was ascending before the news and kept ascending with a rate higher than VAR_MIN = positive
        elif corpus_df.loc[index]["Var%"] > 0 and corpus_df.loc[index + 1]["Var%"] > var:
            sentiment_list.append(POSITIVE_RELEVANCE)

        # if the stoke was descending before the news and kept descending with a rate lower than -VAR_MIN = negative
        elif corpus_df.loc[index]["Var%"] < 0 and corpus_df.loc[index+1]["Var%"] < -var:
            sentiment_list.append(NEGATIVE_RELEVANCE)

        # if the stoke didn't change polarity and is between -VAR_MIN and VAR_MIN = irrelevant
        else:
            sentiment_list.append(IRRELEVANT)
    sentiment_column = pd.DataFrame({"Sentimento": sentiment_list})
    return sentiment_column


def extract_bag_of_words(content_series):
    log.info("------------------------------")
    log.info("Extracting the bag of words")
    log.info("------------------------------")

    # Clear the contents of each news file
    log.info("Cleaning the news files...")
    content_series = content_series.apply(lambda x: basic_cleaning(x))
    content_series = content_series.apply(lambda x: extra_cleaning(x))

    # Remove the portuguese stop words and create a Bag of Words
    stop_words = nltk.corpus.stopwords.words("portuguese")
    cv = CountVectorizer(stop_words=stop_words)
    log.info("Extracting the bag of words...")
    content_series_cv = cv.fit_transform(content_series)
    bag_of_words = pd.DataFrame(content_series_cv.toarray(), columns=cv.get_feature_names())
    bag_of_words = bag_of_words.transpose()
    log.info("Bag of words ready!")
    return bag_of_words


def basic_cleaning(text):
    text = text.lower()
    # Getting rid of symbols and punctuation
    text = re.sub(r"[^0-9a-záéíóúàèìòùâêîôûãõç\s]", "", text)
    # Getting rid of words with numbers in them, or just numbers
    text = re.sub(r"\w*\d\w*", "", text)
    return text


def extra_cleaning(text):
    # There is a session on some texts that just shows some stoke market data, not relevant for our analysis
    text = re.sub(r'DESTAQUES.*', ' ', text)
    return text


def find_new_stop_words(bag_of_words):
    words_freq = {}
    for document in bag_of_words.columns:
        document_words = bag_of_words[document].sort_values(ascending=False)
        words_freq[document] = list(zip(document_words.index, document_words.values))
    data_set_words = []
    for document in bag_of_words.columns:
        all_doc_words = [word for (word, frequency) in words_freq[document] if frequency > 0]
        for word in all_doc_words:
            data_set_words.append(word)
    data_set_word_freq = Counter(data_set_words).most_common()
    # new_stop_words = [word for (word, count) in data_set_word_freq[:30]]
    new_stop_words = [word for (word, count) in data_set_word_freq[:32] if word not in ("queda", "alta")]
    return new_stop_words


def treat_content(text, stop_words):
    text = basic_cleaning(text)
    text = extra_cleaning(text)
    for word in stop_words:
        regex_word = r"\b(" + re.escape(word) + r")\b"
        text = re.sub(regex_word, "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_embedding_layer(text_train, text_test):
    log.info("Preparing the embedding layer")
    # Turn text into Tokens
    tokenizer = Tokenizer(num_words=50000)
    tokenizer.fit_on_texts(text_train)
    text_train = tokenizer.texts_to_sequences(text_train)
    text_test = tokenizer.texts_to_sequences(text_test)
    # Set a maximum sequence for analysis
    vocabulary_size = len(tokenizer.word_index)+1
    text_train = pad_sequences(text_train, padding="post", maxlen=250)
    text_test = pad_sequences(text_test, padding="post", maxlen=250)
    # Get the word's weight with Glove
    embeddings_dict = dict()
    glove_file = open(GLV_FL_PATH, 'r', errors="ignore", encoding="utf8")
    for line in glove_file:
        records = line.split()
        word = records[0]
        try:
            vector_dimensions = asarray(records[1:], dtype="float32")
        except Exception as error:
            log.error(error)
            continue
        embeddings_dict[word] = vector_dimensions
    glove_file.close()
    # Build the embedding matrix using the Glove weights and the tokenized text
    embedding_matrix = zeros((vocabulary_size, 100))
    unused_words = []
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
        else:
            unused_words.append(word)
    pd.DataFrame(unused_words).to_csv("unused_words.csv")
    embedding_layer = Embedding(vocabulary_size, 100, weights=[embedding_matrix], input_length=250, trainable=False)
    return text_train, text_test, embedding_layer


def result_plots(history, var):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(f'Model Accuracy - Irrelevant Variance: {var}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training Data Set', 'Test Data Set'], loc='upper left')
    plt.savefig(f"{GRAPH_PATH}accuracy_var_{var}.png")
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - Irrelevant Variance: {var}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Data Set', 'Test Data Set'], loc='upper left')
    plt.savefig(f"{GRAPH_PATH}loss_var_{var}.png")
    plt.close()


if __name__ == '__main__':
    log.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=log.INFO)
    for variance in range(20):
        log.info("---------------------------------------------------------------------")
        log.info(f"TESTING FOR IRRELEVANT WITH VARIANCE AS: {VAR_MIN*variance}")
        log.info("-----------------------------------------------------------------------")
        main(VAR_MIN*variance)

