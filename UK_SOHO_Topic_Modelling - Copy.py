{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df= pd.read_csv('C:\\\\Users\\\\MalikM\\\\Documents\\\\CVM TEAM -PUNE\\\\UK\\\\CARE COMBINED\\\\input_for_topic_modelling_CustServe.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "len(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df1=df['res']\n",
    "df1=list(df['review'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(df1.dtype)\n",
    "len(df1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#df1.loc[2,]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk\n",
    "from nltk.corpus import stopwords \n",
    "from nltk.stem.wordnet import WordNetLemmatizer\n",
    "import string\n",
    "nltk.download('stopwords')\n",
    "nltk.download('wordnet')\n",
    "\n",
    "#stop = set(stopwords.words('english'))\n",
    "\n",
    "\n",
    "\n",
    "#doc_clean = [clean(doc).split() for doc in doc_complete] \n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "stop = set(stopwords.words('english'))\n",
    "exclude = set(string.punctuation) \n",
    "lemma = WordNetLemmatizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean(doc):\n",
    "     stop_free = \" \".join([i for i in doc.split() if i not in stop])\n",
    "     punc_free = ''.join(ch for ch in stop_free if ch not in exclude)\n",
    "     normalized = \" \".join(lemma.lemmatize(word) for word in punc_free.split())\n",
    "     return normalized"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "nltk.download('wordnet')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df2=pd.DataFrame(df1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "len(df2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df3=df2[~df2['res'].isin(['1','2','3','4','5','6','7','8','9','-',''])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df3_=df3.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df4=df3_[~df3_.res.str.contains(r'[0-9]')]\n",
    "#df4=df3_[df3_.Recommendation.str.isalpha()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df5=list(df4['res'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df5=df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "type(df5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "doc_clean = [clean(doc).split() for doc in df5] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "doc_clean"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing Gensim\n",
    "import gensim\n",
    "from gensim import corpora\n",
    "# Creating the term dictionary of our courpus, where every unique term is assigned an index. \n",
    "dictionary = corpora.Dictionary(doc_clean)\n",
    "\n",
    "# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.\n",
    "doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "doc_term_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating the object for LDA model using gensim library\n",
    "Lda = gensim.models.ldamodel.LdaModel\n",
    "\n",
    " # Running and Trainign LDA model on the document term matrix.\n",
    "ldamodel = Lda(doc_term_matrix, num_topics=4, id2word = dictionary, passes=50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ldamodel.get_topics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(ldamodel.print_topics(num_topics=10, num_words=4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### detailed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import re, numpy as np, pandas as pd\n",
    "from pprint import pprint"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import gensim, spacy, logging, warnings\n",
    "import gensim.corpora as corpora\n",
    "from gensim.utils import lemmatize, simple_preprocess\n",
    "from gensim.models import CoherenceModel\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import stopwords\n",
    "stop_words = stopwords.words('english')\n",
    "# stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])\n",
    "\n",
    "%matplotlib inline\n",
    "warnings.filterwarnings(\"ignore\",category=DeprecationWarning)\n",
    "logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df5[1:4] # sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sent_to_words(sentences):\n",
    "    for sent in sentences:\n",
    "        sent = re.sub('\\S*@\\S*\\s?', '', sent)  # remove emails\n",
    "        sent = re.sub('\\s+', ' ', sent)  # remove newline chars\n",
    "        sent = re.sub(\"\\'\", \"\", sent)  # remove single quotes\n",
    "        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) \n",
    "        yield(sent)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#data = list(df4['Recommendation']).content.values.tolist()\n",
    "data_words = list(sent_to_words(df5))\n",
    "print(data_words[:1])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.\n",
    "trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  \n",
    "bigram_mod = gensim.models.phrases.Phraser(bigram)\n",
    "trigram_mod = gensim.models.phrases.Phraser(trigram)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(bigram)\n",
    "print(trigram)\n",
    "print(bigram_mod)\n",
    "print(trigram_mod)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#print(bigram_mod.common_terms)\n",
    "print(bigram_mod.phrasegrams )\n",
    "print(trigram_mod.phrasegrams )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#nlp = spacy.load('en', disable=['parser', 'ner'])\n",
    "#doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')\n",
    "nlp=spacy.load('en_core_web_sm')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# !python3 -m spacy download en  # run in terminal once\n",
    "def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):\n",
    "    \"\"\"Remove Stopwords, Form Bigrams, Trigrams and Lemmatization\"\"\"\n",
    "    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]\n",
    "    texts = [bigram_mod[doc] for doc in texts]\n",
    "    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]\n",
    "    texts_out = []\n",
    "    #nlp = spacy.load('en', disable=['parser', 'ner'])\n",
    "    for sent in texts:\n",
    "        doc = nlp(\" \".join(sent)) \n",
    "        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])\n",
    "    # remove stopwords once more after lemmatization\n",
    "    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    \n",
    "    return texts_out\n",
    "\n",
    "data_ready = process_words(data_words)  # processed Text Data!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_ready[1:3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Build the topic Model\n",
    "#To build the LDA topic model using LdaModel(), you need the corpus and the dictionary. \n",
    "#Let’s create them first and then build the model. The trained topics (keywords and weights) are printed below as well."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create Dictionary\n",
    "id2word = corpora.Dictionary(data_ready)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "id2word"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create Corpus: Term Document Frequency\n",
    "corpus = [id2word.doc2bow(text) for text in data_ready]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "corpus[0:4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df5[0:4]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build LDA model\n",
    "lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,\n",
    "                                           id2word=id2word,\n",
    "                                           num_topics=3, \n",
    "                                           random_state=100,\n",
    "                                           update_every=1,\n",
    "#                                            chunksize=12,\n",
    "                                           passes=20,\n",
    "                                           alpha='symmetric',\n",
    "                                           iterations=100,\n",
    "                                           per_word_topics=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pprint(lda_model.print_topics())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pprint(lda_model.print_topics())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pprint(lda_model.print_topics())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def format_topics_sentences(ldamodel, corpus, texts):\n",
    "# Init output  \n",
    "    sent_topics_df = pd.DataFrame()\n",
    "\n",
    "# Get main topic in each document\n",
    "    for i, row_list in enumerate(ldamodel[corpus]):\n",
    "        row = row_list[0] if ldamodel.per_word_topics else row_list            \n",
    "        # print(row)\n",
    "        row = sorted(row, key=lambda x: (x[1]), reverse=True)\n",
    "        # Get the Dominant topic, Perc Contribution and Keywords for each document\n",
    "        for j, (topic_num, prop_topic) in enumerate(row):\n",
    "            if j == 0:  # => dominant topic\n",
    "                wp = ldamodel.show_topic(topic_num)\n",
    "                topic_keywords = \", \".join([word for word, prop in wp])\n",
    "                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)\n",
    "            else:\n",
    "                break\n",
    "    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']\n",
    "    # Add original text to the end of the output\n",
    "    contents = pd.Series(texts)\n",
    "    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)\n",
    "    return(sent_topics_df)\n",
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Format\n",
    "df_dominant_topic = df_topic_sents_keywords.reset_index()\n",
    "df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']\n",
    "df_dominant_topic.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##lets see each topic - self\n",
    "topic2=df_dominant_topic[df_dominant_topic[\"Dominant_Topic\"]==2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "topic2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(topic2.loc[[0,32,34,41,65],[\"Text\"]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#7. The most representative sentence for each topic\n",
    "# Display setting to show more characters in column\n",
    "pd.options.display.max_colwidth = 100\n",
    "\n",
    "sent_topics_sorteddf_mallet = pd.DataFrame()\n",
    "sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')\n",
    "\n",
    "for i, grp in sent_topics_outdf_grpd:\n",
    "    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, \n",
    "                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], \n",
    "                                            axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reset Index    \n",
    "sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)\n",
    "\n",
    "# Format\n",
    "sent_topics_sorteddf_mallet.columns = ['Topic_Num', \"Topic_Perc_Contrib\", \"Keywords\", \"Representative Text\"]\n",
    "\n",
    "# Show\n",
    "sent_topics_sorteddf_mallet.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#8. Frequency Distribution of Word Counts in Documents"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "doc_lens = [len(d) for d in df_dominant_topic.Text]\n",
    "\n",
    "# Plot\n",
    "plt.figure(figsize=(16,7), dpi=160)\n",
    "plt.hist(doc_lens, bins = 1000, color='navy')\n",
    "plt.text(750, 100, \"Mean   : \" + str(round(np.mean(doc_lens))))\n",
    "plt.text(750,  90, \"Median : \" + str(round(np.median(doc_lens))))\n",
    "plt.text(750,  80, \"Stdev   : \" + str(round(np.std(doc_lens))))\n",
    "plt.text(750,  70, \"1%ile    : \" + str(round(np.quantile(doc_lens, q=0.01))))\n",
    "plt.text(750,  60, \"99%ile  : \" + str(round(np.quantile(doc_lens, q=0.99))))\n",
    "\n",
    "plt.gca().set(xlim=(0, 200), ylabel='Number of Documents', xlabel='Document Word Count')\n",
    "plt.tick_params(size=16)\n",
    "plt.xticks(np.linspace(0,500,9))\n",
    "plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#9. Word Clouds of Top N Keywords in Each Topic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Wordcloud of Top N words in each topic\n",
    "from matplotlib import pyplot as plt\n",
    "from wordcloud import WordCloud, STOPWORDS\n",
    "import matplotlib.colors as mcolors\n",
    "\n",
    "cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'\n",
    "\n",
    "cloud = WordCloud(stopwords=stop_words,\n",
    "                  background_color='white',\n",
    "                  width=2500,\n",
    "                  height=1800,\n",
    "                  max_words=10,\n",
    "                  colormap='tab10',\n",
    "                  color_func=lambda *args, **kwargs: cols[i],\n",
    "                  prefer_horizontal=1.0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "topics = lda_model.show_topics(formatted=False)\n",
    "fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)\n",
    "for i, ax in enumerate(axes.flatten()):\n",
    "    fig.add_subplot(ax)\n",
    "    topic_words = dict(topics[i][1])\n",
    "    cloud.generate_from_frequencies(topic_words, max_font_size=300)\n",
    "    plt.gca().imshow(cloud)\n",
    "    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))\n",
    "    plt.gca().axis('off')\n",
    "\n",
    "\n",
    "plt.subplots_adjust(wspace=0, hspace=0)\n",
    "plt.axis('off')\n",
    "plt.margins(x=0, y=0)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#10.Let’s plot the word counts and the weights of each keyword in the same chart.\n",
    "from collections import Counter\n",
    "topics = lda_model.show_topics(formatted=False)\n",
    "data_flat = [w for w_list in data_ready for w in w_list]\n",
    "counter = Counter(data_flat)\n",
    "\n",
    "out = []\n",
    "for i, topic in topics:\n",
    "    for word, weight in topic:\n",
    "        out.append([word, i , weight, counter[word]])\n",
    "\n",
    "df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        \n",
    "\n",
    "# Plot Word Count and Weights of Topic Keywords\n",
    "fig, axes = plt.subplots(3, 3, figsize=(16,10), sharey=True, dpi=160)\n",
    "cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]\n",
    "for i, ax in enumerate(axes.flatten()):\n",
    "    ax.bar(x='word', height=\"word_count\", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')\n",
    "    ax_twin = ax.twinx()\n",
    "    ax_twin.bar(x='word', height=\"importance\", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')\n",
    "    ax.set_ylabel('Word Count', color=cols[i])\n",
    "    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)\n",
    "    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)\n",
    "    ax.tick_params(axis='y', left=False)\n",
    "    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')\n",
    "    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')\n",
    "\n",
    "fig.tight_layout(w_pad=2)    \n",
    "fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#14.Finally, pyLDAVis is the most commonly used and a nice way to visualise \n",
    "#the information contained in a topic model. Below is the implementation for LdaModel()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pyLDAvis.gensim\n",
    "pyLDAvis.enable_notebook()\n",
    "vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)\n",
    "vis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### https://remicnrd.github.io/Aspect-based-sentiment-analysis/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df4_1=pd.read_csv(\"C:\\\\Users\\\\MalikM\\\\Documents\\\\exit interview\\\\topic modelling\\\\Exit_Interviews_Recommendations_TM_1.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv(\"C:\\\\Users\\\\MalikM\\\\Documents\\\\exit interview\\\\aug\\\\Key_phrases_exit_int__eng_req_aug_ver1.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df4_1.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df4_2=df4_1[[\"Recommendation\",'aspect_category', 'sentiment']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_1=df[[\"recommendations\",'Main_Phrase']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2 = df_1.rename(columns = {\"Main_Phrase\": \"aspect_category\"}) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# aspect based"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import spacy\n",
    "#nlp = spacy.load('en', disable=['parser', 'ner'])\n",
    "#doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')\n",
    "nlp=spacy.load('en_core_web_sm')\n",
    "# nlp = spacy.load('en')\n",
    "\n",
    "#dataset.review=df4_2.Recommendation.str.lower()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df_2[\"review\"] = df_2.recommendations.str.lower()\n",
    "#df4_2.review=df4_2[~df4_2['review'].isin(['1','2','3','4','5','6','7','8','9','-',''])]\n",
    "#df4_2.review=df4_2[~df4_2.review.str.contains(r'[0-9]')]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2=df_2[~df_2['review'].isin(['1','2','3','4','5','6','7','8','9','-',''])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df_2=df_2.dropna()\n",
    "df_2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "aspect_terms = []\n",
    "for review in nlp.pipe(df_2.review):\n",
    "    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN'] # taking out noun terms\n",
    "    aspect_terms.append(' '.join(chunks))\n",
    "df_2['aspect_terms'] = aspect_terms\n",
    "df_2.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "from keras.models import load_model\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Activation\n",
    "\n",
    "aspect_categories_model = Sequential()\n",
    "aspect_categories_model.add(Dense(512, input_shape=(6000,), activation='relu'))\n",
    "aspect_categories_model.add(Dense(10, activation='softmax'))\n",
    "aspect_categories_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.preprocessing.text import Tokenizer\n",
    "\n",
    "vocab_size = 6000 # We set a maximum size for the vocabulary\n",
    "tokenizer = Tokenizer(num_words=vocab_size)\n",
    "tokenizer.fit_on_texts(df_2.review)\n",
    "aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df_2.aspect_terms))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "from keras.utils import to_categorical\n",
    "\n",
    "label_encoder = LabelEncoder()\n",
    "integer_category = label_encoder.fit_transform(df_2.aspect_category)\n",
    "dummy_category = to_categorical(integer_category)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2.aspect_terms.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dummy_category"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "integer_category\n",
    "#dummy_category.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2.aspect_category.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "aspect_categories_model.fit(aspect_tokenized, dummy_category, epochs=5, verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Let’s try our Aspect Categories model on a new review. We have to do the same processing \n",
    "#operations on that review and then predict using model.predict. And not forget to transform the output using "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# new_review = \"The higher management is biased and gives less inreament\"\n",
    "# new_review= 'Yearly Appraisals can be more attractive with more official facilities, new technological trainings can be provided to the employees.'\n",
    "new_review='I tried to share my views with higher authorities for disputes with team leader but nothing changed, infect my team leader came to me and said that I tried with higher authority for disputes with him but nothing will change.. This is how a agent level person is facing issues while working in Vodafone, so please try to change this for betterment of lower level employees. Thank you.'\n",
    "chunks = [(chunk.root.text) for chunk in nlp(new_review).noun_chunks if chunk.root.pos_ == 'NOUN']\n",
    "new_review_aspect_terms = ' '.join(chunks)\n",
    "new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])\n",
    "\n",
    "new_review_category = label_encoder.inverse_transform(aspect_categories_model.predict_classes(new_review_aspect_tokenized))\n",
    "print(new_review_category)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Step 3: Get the Sentiment Terms -not required for now"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Now, we want to keep from our reviews only the words which can indicate if the expressed opinion is positive or negative.\n",
    "\n",
    "#We use spaCy once again, but this time we are going to use Part Of Speech Tagging to filter and keep only adjectives and verbs.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# sentiment_terms = []\n",
    "# for review in nlp.pipe(df4_2['review']):\n",
    "#         if review.is_parsed:\n",
    "#             sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == \"ADJ\" or token.pos_ == \"VERB\"))]))\n",
    "#         else:\n",
    "#             sentiment_terms.append('')  \n",
    "# df4_2['sentiment_terms'] = sentiment_terms\n",
    "# df4_2.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Step 4: Build the Sentiment Model\n",
    "\n",
    "#For the sentiment model, we use a very similar architecture than in step 2:\n",
    "#•A dense layer that take as input word vectors of dimension 6000, with a relu activation function and 512 nodes.\n",
    "#•An output layer with a softmax activation function to output probability distribution. This time with only 3 nodes because we want to predict positive, negative or neutral.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# sentiment_model = Sequential()\n",
    "# sentiment_model.add(Dense(512, input_shape=(6000,), activation='relu'))\n",
    "# sentiment_model.add(Dense(3, activation='softmax'))\n",
    "# sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#The input is going to be the feature sentiment_terms, but we need to use the same encoding process as for the aspect terms. \n",
    "#This time we don’t have to create and fit a Tokenizer, because it already has been done on the entire review dataset in step 2."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# sentiment_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df4_2.sentiment_terms))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#The output is the feature sentiment, which requires an encoding to dummy variable. We need to create a new labelEncoder because we have only 3 categories,\n",
    "#which is different from the 12 aspect categories."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# label_encoder_2 = LabelEncoder()\n",
    "# integer_sentiment = label_encoder_2.fit_transform(df4_2.sentiment)\n",
    "# dummy_sentiment = to_categorical(integer_sentiment)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df4_2.sentiment.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# sentiment_model.fit(sentiment_tokenized, dummy_sentiment, epochs=5, verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# new_review = \"This italian place is nice and cosy\"\n",
    "\n",
    "# chunks = [(chunk.root.text) for chunk in nlp(new_review).noun_chunks if chunk.root.pos_ == 'NOUN']\n",
    "# new_review_aspect_terms = ' '.join(chunks)\n",
    "# new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])\n",
    "\n",
    "# new_review_category = label_encoder_2.inverse_transform(sentiment_model.predict_classes(new_review_aspect_tokenized))\n",
    "# print(new_review_category)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_reviews = [\"Vodafone should consider his sales employees as individuals not just numbers\",\n",
    "                \"We need to a have a stronger support system in place for resolving customer issues\",\n",
    "                \"Well, it is a big job, the process in customer experience should be much clearer, predictable, more consistent\",\n",
    "                \"better management and hr and salary and growth to work with\",\n",
    "                \"A very solid succession plan that is communicated to the person in question.\",\n",
    "                \"Stop growing Politics and HSW policy is way too much strict no job security so i think it should get revised\"]\n",
    "\n",
    "# Aspect preprocessing\n",
    "test_reviews = [review.lower() for review in test_reviews]\n",
    "test_aspect_terms = []\n",
    "for review in nlp.pipe(test_reviews):\n",
    "    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']\n",
    "    test_aspect_terms.append(' '.join(chunks))\n",
    "test_aspect_terms = pd.DataFrame(tokenizer.texts_to_matrix(test_aspect_terms))\n",
    "                             \n",
    "# Sentiment preprocessing\n",
    "# test_sentiment_terms = []\n",
    "# for review in nlp.pipe(test_reviews):\n",
    "#         if review.is_parsed:\n",
    "#             test_sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == \"ADJ\" or token.pos_ == \"VERB\"))]))\n",
    "#         else:\n",
    "#             test_sentiment_terms.append('') \n",
    "# test_sentiment_terms = pd.DataFrame(tokenizer.texts_to_matrix(test_sentiment_terms))\n",
    "\n",
    "# Models output\n",
    "test_aspect_categories = label_encoder.inverse_transform(aspect_categories_model.predict_classes(test_aspect_terms))\n",
    "# test_sentiment = label_encoder_2.inverse_transform(sentiment_model.predict_classes(test_sentiment_terms))\n",
    "for i in range(5):\n",
    "#     print(\"Review \" + str(i+1) + \" is expressing a  \" + test_sentiment[i] + \" opinion about \" + test_aspect_categories[i])\n",
    "    print(\"Review \" + str(i+1) + \" is expressing a  opinion about \" + test_aspect_categories[i])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "abc=\"Vodafone should consider his sales .employees as .individuals not just numbers\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sen=pd.DataFrame(abc.split('.'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sen"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ata={ 'name':['mukul','aakash'], 'comments':['yes sir. ok sir','gd ha,land nahi']}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.DataFrame(ata)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['new']=data['comments'].split(.)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
