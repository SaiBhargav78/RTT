import spacy
import pytextrank
import re

nlp = spacy.load("en_core_web_sm") # or: de_core_news_sm

# Add the textrank algorithm to the spacy pipeline
nlp.add_pipe("textrank")

def summarize(text):
# We want to cleanup the text a bit.
    text = text.replace("\n", "").replace("\r", "")
    text = re.sub(' +', ' ', text)
    # The doc now contains the summary. That is it!
    doc = nlp(text)
    resulting_sentences = 5
    res = "".join(token.text+"\n" for token in doc._.textrank.summary(limit_sentences=resulting_sentences))
    # Print out the results
    # for sentence in doc._.textrank.summary(limit_sentences=resulting_sentences):
    #     res+=sentence
    return res
