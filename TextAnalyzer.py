import math
import os
import string
from collections import Counter
from sankeystuff import *
import pandas as pd
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import scattertext as st
from Gutenberg_Parser import *
from textblob import TextBlob
import spacy
import numpy as np



# TextAnalyzer class accepts file_info dictionary of labels and filepaths of relevant texts
# Also includes useful methods for text visualization.
class TextAnalyzer():
    def __init__(self, file_info, stopfilepath):
        self.file_info = file_info
        self.stop_words = self.load_stop_words(stopfilepath)
        self.dict_texts = self.load_files(parser=gutenberg_parser)
        self.word_counts = self.count_words()
        self.df_word_counts = self.word_count_transform()
        self.df_sentiment = self.sentiment()


    # accept a word count dictionary and return a dataframe with words as rows and their word counts as data points
    def word_count_transform(self):
        list_df = []
        for key in self.word_counts:
            df_key = pd.DataFrame(data=self.word_counts[key], index=['Word Count']).T
            df_key['Book'] = key
            df_key = df_key.sort_values(by='Word Count', ascending=False)
            list_df.append(df_key)

        df_word_count = pd.concat(list_df, axis=0)
        df_word_count.reset_index(inplace=True)
        return df_word_count



    # given a list of lists of tokenized and cleaned text, return word counts
    def count_words(self):
        dict_word_counts = {}
        for key in self.dict_texts:

            word_counts = Counter(self.dict_texts[key][0])
            dict_word_counts[key] = dict(word_counts)

        return dict_word_counts

    # accept raw text and return a list of words cleaned of punctuation and stop words
    def clean_text(self, raw_text):
        # convert to lower case
        tokens = [w.lower() for w in raw_text]
        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]
        # filter out stop words
        words = [w for w in words if w not in self.stop_words]
        return words

    # accept a filename, labels, and a parser and return list of cleaned words
    def load_text(self, filename, label="", parser=None):
        if parser is None:
            with open(filename, 'r') as f:
                raw_text = f.read()
                text = self.clean_text(raw_text)
            return text
        else:
            text = parser(filename)
            text = self.clean_text(text)
            return text

    # given filename from class instance iterate thru files and create dictionary of each text
    def load_files(self,label = "",parser = None):
        dict_texts = {}
        for key in self.file_info:
            directory = self.file_info[key]
            os.chdir(directory)
            # iterate through all file
            for file in os.listdir():
                list_texts = []
            # Check whether file is in text format or n==----
                if file.endswith(".txt"):
                    filename = f"{directory}/{file}"
                    # call read text file function
                    text = self.load_text(filename, label, parser)
                    list_texts.append(text)
                    # column counts for each text
                    dict_texts[file[:-4]] = list_texts
                    # need to find a way to concatenate names? having challenges here.
        #print(dict_texts)
        return dict_texts

    # load stop words given a file name
    @staticmethod
    def load_stop_words(stopfile):
        with open(stopfile, 'r') as f:
            stopwords = f.read()
            stopwords = stopwords.split('\n')
        return stopwords

    # given a dictionary of word counts display a sankey diagram based of of params
    def wordcount_sankey(self, word_list=None, k=5):
        # making sankey based on top 5 words in each document
        allwords = []

        if word_list == None:
            for i in self.dict_texts.keys():
                #print(i)
                new = self.df_word_counts[self.df_word_counts['Book'] == str(i)].sort_values(by='Word Count', ascending=False)
                new = new.head(k)
                #print(new)
                new = new['index'].tolist()
                for i in new:
                    allwords.append(i)
            df = self.df_word_counts[self.df_word_counts['index'].isin(allwords)]
            make_sankey(df, src='Book', targ='index', vals='Word Count')
        # making sankey based on what user searches
        else:
            # essentially just generating dataframe for each word in wordlist, unsure if more efficient way but this is what i'm doing.
            df = self.df_word_counts[self.df_word_counts['index'].isin(word_list)]
            make_sankey(df, src='Book', targ='index', vals='Word Count')

    # save a png file of a given instance's dictionary of texts
    def wordcloud(self,masks = None):
        """Takes a folder of Masks (Jpegs/pngs) with same naming conventions
        as novel text file and creates wordclouds with mask on it
        """
        num = 1
        fig = plt.figure(figsize=(24,14),dpi=1200)
        if masks == None:
            for i in self.dict_texts.keys():
                valstring = (",".join([str(i) for i in self.dict_texts[i]]))
                ax = fig.add_subplot(2,math.ceil(len(self.dict_texts.keys())/2),num)
                ax.set_title(i)
                wordcloud = WordCloud().generate(valstring)
                ax.imshow(wordcloud, interpolation = 'bilinear')
                ax.axis('off')
                num += 1
        else:
            for root, dir, files in os.walk(masks):
                for i in self.dict_texts.keys():
                    if (i+'.jpeg') in files:
                        maskin = np.array(Image.open(str(root+'/'+i+'.jpeg')))
                        valstring = (",".join([str(i) for i in self.dict_texts[i]]))
                        ax = fig.add_subplot(2,math.ceil(len(self.dict_texts.keys())/2),num)
                        ax.set_title(i)
                        wordcloud = WordCloud(mask=maskin, background_color='white',contour_width=3, contour_color='black').generate(valstring)
                        ax.imshow(wordcloud, interpolation = 'bilinear')
                        ax.axis('off')
                        num += 1
        fig.tight_layout()
        fig.savefig('example.png')

    # iterate through each text and return a dataframe including book title, full text, and sentiment score
    def sentiment(self):
        """Calculate the sentiment of the text"""
        list_books = []
        list_texts = []
        list_sentiment = []
        total = 0
        total_word = 0
        for key in self.dict_texts:
            text = self.dict_texts[key][0]
            #print(text)
            sentence = ''
            for w in text:
                sentence = sentence + ' ' + w
            pol = TextBlob(sentence).sentiment.polarity
            total += pol
            total_word += 1
            list_books.append(key)
            list_texts.append(sentence)
            list_sentiment.append(pol)
        dict_sent_analysis = {'Book': list_books,
                              'Full Text': list_texts,
                              'Sentiment': list_sentiment}
        df_sentiment = pd.DataFrame(data=dict_sent_analysis)
        return df_sentiment

    # save html file of scattertext from outside library
    def scatterxt(self):
        df = self.df_sentiment

        df.plot.bar(x='Book',y='Sentiment')
        plt.title('Sentiment Per Text')
        plt.xticks(rotation=45, horizontalalignment="center")
        plt.tight_layout()
        plt.savefig('sentiment.png')

        df['PosNeg'] = 'Positive'
        df.loc[df['Sentiment'] < 0, 'PosNeg'] = 'Negative'
        df.loc
        nlp = spacy.load("en_core_web_sm")
        corpus = (st.CorpusFromPandas(df, category_col='Book', text_col='Full Text', nlp=nlp).build())
        html = st.produce_scattertext_explorer(
            corpus,
            category='Positive', category_name='Positive', not_category_name='Negative'
            , metadata=corpus.get_df()['Book']
            )
        open("Sentiment_Analysis.html", 'wb').write(html.encode('utf-8'))


# file paths of stop words and book files
stopfile = 'https://github.com/eliaskarikas/NLPLibrary-GutenbergProcessing/blob/main/stopwords.txt'
dict_filepath = {'https://github.com/eliaskarikas/NLPLibrary-GutenbergProcessing/tree/main/books'
                 }

if __name__ == '__main__':
    monster_text = TextAnalyzer(dict_filepath, stopfile)
    monster_text.scatterxt()
    monster_text.wordcount_sankey(word_list=['love','beautiful','romantic','emotion','evil','death','kill','dark'])
    masks='https://github.com/eliaskarikas/NLPLibrary-GutenbergProcessing/tree/main/books/masks'
    monster_text.wordcloud(masks=masks)






