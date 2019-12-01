import logging as log
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from scraper.source.BrInvestScraper import search_for_links as scrape
from scraper.source.BrInvestScraper import get_link_content as get_content
from sentiment.source.SentimentNetwork import extract_bag_of_words
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
import webbrowser
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import os


class StokeDB(BoxLayout):
    stoke_alias = ObjectProperty()
    analysis_day = ObjectProperty()
    relevant_news = ObjectProperty()
    last_news = scrape(depth=1, app=True)
    tokenizer = Tokenizer(num_words=1000)

    def submit_analysis(self, submitted_date):
        if self.stoke_alias != "PETR3":
            return
        results = []
        for news_link in self.last_news:
            content = get_content(news_link)
            news_date = content[1].split("¨")[1]
            if news_date == submitted_date:
                news = self.tokenizer.texts_to_sequences(content[1].split("¨")[2])
                n_list = []
                for sublist in news:
                    for item in sublist:
                        n_list.append(item)
                n_list = [n_list]
                news = pad_sequences(n_list, padding="post", maxlen=100)
                sentiment = App.get_running_app().model.predict(news)
                results.append(sentiment)
        self.update_the_variables(float(sum(results)))

    def update_the_variables(self, final_decision):
        self.last1.text = self.last_news[0]
        self.last2.text = self.last_news[1]
        self.last3.text = self.last_news[2]
        self.last4.text = self.last_news[3]
        self.last5.text = self.last_news[4]
        self.slider.value = final_decision

    def reset_analysis(self):
        self.last1.text = ""
        self.last2.text = ""
        self.last3.text = ""
        self.last4.text = ""
        self.last5.text = ""
        self.slider.value = 0

    def redirect_to_news(self, link):
        webbrowser.open(link)

    def spinner_clicked(self, value):
        self.stoke_alias = value


class StokeDBApp(App):
    link1 = ""
    link2 = ""
    link3 = ""
    link4 = ""
    link5 = ""
    final_decision = 0
    model = load_model(f"{os.path.dirname(os.path.abspath(__file__))}\\..\\..\\sentiment\\SentimentModel.h5")

    def build(self):
        return StokeDB()

