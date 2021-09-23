{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy\n",
    "#nlp = spacy.load('en', disable=['parser', 'ner'])\n",
    "#doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')\n",
    "nlp=spacy.load('en_core_web_sm')\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from textblob import TextBlob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df4_1=pd.read_csv(\"C:\\\\Users\\\\MalikM\\\\Documents\\\\CVM TEAM -PUNE\\\\UK\\\\BB Combined\\\\SOHO Broadband May_June.csv\",encoding='ISO-8859-1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2=pd.DataFrame(df4_1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2.head(10)"
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
    "df_2[\"review\"] = df_2['Verbatim'].str.lower()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2['Themes '].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2['Themes '].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2['Themes '][df_2['Themes '].isnull()]='Uncategorised' # replacing Blanks with Uncategorised"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2['Themes '].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#### taking Uncategorised and COVID-19 out for testing as they are wrongly classified ###"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "train=df_2[~((df_2['Themes ']=='Uncategorised') | (df_2['Themes ']=='COVID-19') | (df_2['Themes '].isnull()))]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "test=df_2[(df_2['Themes ']=='Uncategorised') | (df_2['Themes ']=='COVID-19') | (df_2['Themes '].isnull())]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train['Themes '].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test['Themes '].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2_=train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "df_2_=df_2_[~df_2_['review'].isin(['1','2','3','4','5','6','7','8','9','-',''])]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2_.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "df_2[['Verbatim','review']].head(10)"
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
    "\n",
    "# df_3=df_2['review'].dropna()\n",
    "df_3=df_2_[~(df_2_['review'].isnull())]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_3.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df_3.reset_index(drop=True,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_3[['Verbatim','review']].head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df_3['review']=df_3['review'].str.extract('([a-zA-Z ]+)', expand=False).str.strip()\n",
    "# df_3.head(12)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df_3[['Verbatim','review']].head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    " df_3.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    " # NOUNS for aspect"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# taking out noun terms\n",
    "\n",
    "\n",
    "aspect_terms = []\n",
    "for review in nlp.pipe(df_3.review):\n",
    "    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN'] \n",
    "    aspect_terms.append(' '.join(chunks))\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "df_3['aspect_terms'] = aspect_terms\n",
    "df_3.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## innitialinsing CNN\n",
    "from keras.models import load_model\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Activation\n",
    "\n",
    "aspect_categories_model = Sequential()\n",
    "aspect_categories_model.add(Dense(512, input_shape=(6000,), activation='relu'))\n",
    "aspect_categories_model.add(Dense(7, activation='softmax'))\n",
    "aspect_categories_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])"
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
    "tokenizer.fit_on_texts(df_3.review)\n",
    "aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df_3.aspect_terms))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Label encoding the aspect category\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from keras.utils import to_categorical\n",
    "\n",
    "label_encoder = LabelEncoder()\n",
    "integer_category = label_encoder.fit_transform(df_3['Themes '])\n",
    "dummy_category = to_categorical(integer_category)"
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
    "df_3['Themes '].unique()"
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
    "#  taking out sentiment terms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from textblob import TextBlob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "df_3['polarity']=df_3.apply(lambda x : TextBlob(x['review']).sentiment.polarity, axis=1)\n",
    "df_3['subjectivity']=df_3.apply(lambda x : TextBlob(x['review']).sentiment.subjectivity,axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_3.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "df_3['sentiment']=np.where(df_3['polarity']<0,'Negative','Neutral')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df_3"
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
   "source": []
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
   "source": []
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
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sentiment_terms = []\n",
    "for review in nlp.pipe(df_3['review']):\n",
    "        if review.is_parsed:\n",
    "            sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == \"ADJ\" or token.pos_ == \"VERB\"))]))\n",
    "        else:\n",
    "            sentiment_terms.append('')  \n",
    "df_3['sentiment_terms'] = sentiment_terms\n",
    "df_3.head(10)"
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
    "# sentiment_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(df_3.sentiment_terms))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#creating label encoder\n",
    "# label_encoder_2 = LabelEncoder()\n",
    "# integer_sentiment = label_encoder_2.fit_transform(df_3.sentiment)\n",
    "# dummy_sentiment = to_categorical(integer_sentiment)"
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
    "#taking train output for aspect terms, sentiment terms, and sentiment\n",
    "df_3.to_csv(\"C:\\\\Users\\\\MalikM\\\\Documents\\\\CVM TEAM -PUNE\\\\UK\\\\BB Combined\\\\Train_UK_SOHO_ABSA_bb_May_June_OUTPUT.csv\")"
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
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#testing on test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df_test2_=test[~test['review'].isin(['1','2','3','4','5','6','7','8','9','','Na','-','.'])]\n",
    "# df_test2_opposite=test[test['review'].isin(['1','2','3','4','5','6','7','8','9','','Na','-','.'])]\n",
    "\n",
    "\n",
    "df_test2__=df_test2_[~df_test2_.review.str.contains(r'\\d',na=True)] # only non nulls\n",
    "# nulls=df_test2_[df_test2_.review.str.contains(r'\\d',na=True)] # only nulls\n",
    "\n",
    "df_test2=df_test2__.replace({'review': {'-': '', '&': 'and', ' e.g. ':' that is ',' e.g ':' that is ',' i.e ': ' that is ','Na':'Nothing',pd.np.nan: 'None' }}, regex=True)\n",
    "# df_test2.to_csv('C:\\\\Users\\\\MalikM\\\\Documents\\\\exit interview\\\\DEC\\\\not_null.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test2__.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test2.reset_index(inplace=True,drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test2.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "count=0\n",
    "aspect=[]\n",
    "aspect_terms=[]\n",
    "for i in df_test2.index:\n",
    "# while i <86:\n",
    "# for i in range(86):\n",
    "# for i in range(85):\n",
    "    \n",
    "#     dd=pd.DataFrame()\n",
    "#     count=count+1\n",
    "    new_review=df_test2['review'][i]\n",
    "#     print(new_review)\n",
    "\n",
    "#     new_review='I tried to share my views with higher authorities for disputes with team leader but nothing changed, infect my team leader came to me and said that I tried with higher authority for disputes with him but nothing will change.. This is how a agent level person is facing issues while working in Vodafone, so please try to change this for betterment of lower level employees. Thank you.'\n",
    "    chunks = [(chunk.root.text) for chunk in nlp(new_review).noun_chunks if chunk.root.pos_ == 'NOUN']\n",
    "\n",
    "    #to take out aspect terms as well\n",
    "    aspect_terms.append(' '.join(chunks))\n",
    "    \n",
    "    new_review_aspect_terms = ' '.join(chunks)\n",
    "    new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])\n",
    "\n",
    "    new_review_category = label_encoder.inverse_transform(aspect_categories_model.predict_classes(new_review_aspect_tokenized))\n",
    "#     print(count)\n",
    "#     print(new_review_category)\n",
    "    aspect.append(new_review_category)\n",
    "    \n",
    "     #df_test4['aspect'][i]=pd.Series(new_review_category)\n",
    "    \n",
    "df_test2['aspect']=pd.Series(aspect) \n",
    "df_test2['aspect_terms']=aspect_terms \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## ----------- no need to make sentiment model as sentiment can be found without any model, with above used code for polarity \n",
    "# sentiment = []\n",
    "# for review in nlp.pipe(df_test2['review']):\n",
    "#     if review.is_parsed:\n",
    "#         sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == \"ADJ\" or token.pos_ == \"VERB\"))]))\n",
    "#     else:\n",
    "#         sentiment_terms.append('') \n",
    "#     test_sentiment_terms_tokenized = tokenizer.texts_to_matrix([test_sentiment_terms])\n",
    "#     new_sentiment_category = label_encoder_2.inverse_transform(sentiment_model.predict_classes(test_sentiment_terms_tokenized))\n",
    "#     sentiment.append(new_sentiment_category)\n",
    "# df_test2['sentiment']=pd.Series(sentiment)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df_test2['polarity']=df_test2.apply(lambda x : TextBlob(x['review']).sentiment.polarity, axis=1)\n",
    "df_test2['subjectivity']=df_test2.apply(lambda x : TextBlob(x['review']).sentiment.subjectivity,axis=1)\n",
    "df_test2['sentiment']=np.where(df_test2['polarity']<0,'Negative', 'Neutral')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sentiment_terms = []\n",
    "for review in nlp.pipe(df_test2['review']):\n",
    "        if review.is_parsed:\n",
    "            sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == \"ADJ\" or token.pos_ == \"VERB\"))]))\n",
    "        else:\n",
    "            sentiment_terms.append('')  \n",
    "df_test2['sentiment_terms'] = sentiment_terms\n",
    "df_test2.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test2.to_csv(\"C:\\\\Users\\\\MalikM\\\\Documents\\\\CVM TEAM -PUNE\\\\UK\\\\BB Combined\\\\Test_UK_SOHO_ABSA_bb_May_June_OUTPUT.csv\")"
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
    "#################################################################################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Train+test -- before splitting\n",
    "df_2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2['Themes '].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_Broadband=df_2[df_2['Themes ']=='Broadband']\n",
    "all_Billing=df_2[df_2['Themes ']=='Billing']\n",
    "all_Customer_Service=df_2[df_2['Themes ']=='Customer Service']\n",
    "all_Technical=df_2[df_2['Themes ']=='Technical']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_Customer_Service_=' '.join(all_Customer_Service['review'].astype(str))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "all_Customer_Service_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk\n",
    "Tokens = nltk.word_tokenize(all_Broadband_)\n",
    "output = list(nltk.bigrams(Tokens))\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "Tokens = nltk.word_tokenize(all)\n",
    "output = list(nltk.trigrams(Tokens))\n",
    "print(output)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tri_freq=nltk.FreqDist(output)\n",
    "tri_freq.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# all_1='Service on the phone was absolutely terrible the girl could hardly speak English and was totally unhelpful and was cause I wanted to see what I had been charged for on my bill'\n",
    "tokens = nltk.word_tokenize(all_Customer_Service_)\n",
    "# print(tokens)\n",
    "tag = nltk.pos_tag(tokens)\n",
    "grammar = r\"\"\"\n",
    "  NP: {<NN.?>*<VBD.?>*<RB.?>*<VB.?>*<JJ.?>*<NN.?>*<NN.?>?}\n",
    "  }<MD>{\"\"\"\n",
    "cp  =nltk.RegexpParser(grammar)\n",
    "result = cp.parse(tag)\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "result_=pd.Series(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "result_df=pd.DataFrame({'res':result_})\n",
    "result_df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "result_df=result_df.replace(['\\)','\\(','\\'',',','/','\\n','NN','VBD','RB','JJ','CC','NP','IN','DT','VBN','VBG','MD','VB','PRP$','S'],'',regex=True).astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "result_df['totalwords'] = result_df['res'].str.split().str.len()\n",
    "# result_df['totalwords'] = df['col'].str.count(' ') + 1\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "result_df=result_df.sort_values('totalwords', ascending=False)\n",
    "# result_df_final=pd.DaraFrame(result_df.res)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "result_df=result_df.replace(['\\)','\\(','\\'',',','/','\\n','P','NN','VBD','RB','JJ','CC','NP','IN','DT','VBN','VBG','MD','VB','PRP$','S'],'',regex=True).astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "result_df.totalwords.astype(int)\n",
    "result_df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# result_df_final.to_csv('C:\\\\Users\\\\MalikM\\\\Documents\\\\CVM TEAM -PUNE\\\\UK\\\\OUTPUT\\\\RegExp_chunk_SOHO_May.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# mylist=['broadband','problem','internet'] # from john - most frequent words used for broadband themes\n",
    "# mylist=['billing','bill'] # from john - most frequent words used for broadband themes\n",
    "mylist=['call','time','call','service','vodafone','broadband','wait','resolve','customer','bussiness'] # from john - most frequent words used for broadband themes\n",
    "pattern = '|'.join(mylist)"
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
    "# these most frequent words are the searched in the Regex-chunked broadband comments.\n",
    "# result_df_billing=result_df[result_df['res'].str.contains(\"bill\") & (result_df['totalwords']>2)]\n",
    "# result_df_broadband=result_df[result_df['res'].str.contains(pattern) & (result_df['totalwords']>1)]\n",
    "# result_df_billing=result_df[result_df['res'].str.contains(pattern) &(result_df.totalwords.astype(int)> 1)]\n",
    "\n",
    "result_df_CustService=result_df[result_df['res'].str.contains(pattern) &(result_df.totalwords.astype(int)> 1)]"
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
    "result_df_CustService.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "result_df_CustService"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "result_df_CustService.to_csv('C:\\\\Users\\\\MalikM\\\\Documents\\\\CVM TEAM -PUNE\\\\UK\\\\OUTPUT\\\\RegExp_chunk_CustService_freq_words_SOHO_May.csv')"
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
   "source": []
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
   "source": []
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
    "###https://remicnrd.github.io/Aspect-based-sentiment-analysis/"
   ]
  },
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
    "df4_1=pd.read_csv(\"C:\\\\Users\\\\MalikM\\\\Documents\\\\CVM TEAM -PUNE\\\\UK\\\\SOHO Care detractors May.csv\",encoding='ISO-8859-1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df4_1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df4_1['aspect_category'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy\n",
    "#nlp = spacy.load('en', disable=['parser', 'ner'])\n",
    "#doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')\n",
    "nlp=spacy.load('en_core_web_sm')\n",
    "# nlp = spacy.load('en')\n",
    "\n",
    "#dataset.review=df4_2.Recommendation.str.lower()"
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
    "df_2=df4_1[[\"recommendations\",'aspect_category']]"
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
    "# import spacy\n",
    "# #nlp = spacy.load('en', disable=['parser', 'ner'])\n",
    "# #doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')\n",
    "# nlp=spacy.load('en_core_web_sm')\n",
    "# nlp = spacy.load('en')\n",
    "\n",
    "#dataset.review=df4_2.Recommendation.str.lower()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Converting to lower string\n",
    "df_2[\"review\"] = df_2.recommendations.str.lower()\n",
    "#df4_2.review=df4_2[~df4_2['review'].isin(['1','2','3','4','5','6','7','8','9','-',''])]\n",
    "#df4_2.review=df4_2[~df4_2.review.str.contains(r'[0-9]')]"
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
    "df_2.head()"
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
    "df_2.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## dropping NA\n",
    "df_3=df_2.dropna()\n",
    "df_3.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_3.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df_3.reset_index(drop=True,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import nltk as nltk\n",
    "# import pandas as pd\n",
    "# import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# s=df_3[\"review\"].str.split('.').apply(pd.Series).stack()\n",
    "# s.index = s.index.droplevel(-1)\n",
    "# s.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# s.to_csv('C:\\\\Users\\\\MalikM\\\\Documents\\\\exit interview\\\\topic modelling\\\\b.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# s=df_3[\"review\"].str.split('.').apply(pd.Series).stack()\n",
    "# s.index = s.index.droplevel(-1) # to line up with df's index\n",
    "# # print(s)\n",
    "# s.name='Comments' # it gives name to series ... for joining as column below. After joining with dataframe- my_df, \n",
    "# #                             this will become the name of the column\n",
    "# s.replace('',np.nan,inplace=True)\n",
    "# s.dropna(inplace=True)\n",
    "# del my_df['comments']\n",
    "# my_df=my_df.join(s)\n",
    "# my_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df_2['review'] = \n",
    "# df_3['review'].astype(str).str.extract('(\\d+)')#.astype(str)\n",
    "\n",
    "df_3['review'].str.extract('([a-zA-Z ]+)', expand=False).str.strip()\n",
    "df_3.head(12)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "df_2.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### try on new data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test=pd.read_csv(\"C:\\\\Users\\\\MalikM\\\\Documents\\\\exit interview\\\\DEC\\\\demo1.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test.head()\n",
    "#print(len(df_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_test1=df_test[[\"DummyID\",\"Response ID\",\"Recommendation\"]]\n",
    "len(df_test1)\n",
    "df_test1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#removing sentience which have only numbers in it\n",
    "\n",
    "df_test2_=df_test1[~df_test1['Recommendation'].isin(['1','2','3','4','5','6','7','8','9','','Na','-','.'])]\n",
    "df_test2_opposite=df_test1[df_test1['Recommendation'].isin(['1','2','3','4','5','6','7','8','9','','Na','-','.'])]\n",
    "\n",
    "\n",
    "df_test2__=df_test2_[~df_test2_.Recommendation.str.contains(r'\\d',na=True)] # only non nulls\n",
    "nulls=df_test2_[df_test2_.Recommendation.str.contains(r'\\d',na=True)] # only nulls\n",
    "\n",
    "df_test2=df_test2__.replace({'Recommendation': {'-': '', '&': 'and', ' e.g. ':' that is ',' e.g ':' that is ',' i.e ': ' that is ','Na':'Nothing',pd.np.nan: 'None' }}, regex=True)\n",
    "# df_test2.to_csv('C:\\\\Users\\\\MalikM\\\\Documents\\\\exit interview\\\\DEC\\\\not_null.csv')\n",
    "\n",
    "\n",
    "# nulls.to_csv('C:\\\\Users\\\\MalikM\\\\Documents\\\\exit interview\\\\DEC\\\\null.csv')\n",
    "\n",
    "# df_test3=df_test2.dropna()\n",
    "# print(len(df_test2))\n",
    "# print(df_test2)\n",
    "\n",
    "\n",
    "# df_test3[df_test3['Recommendation'].str.extract('([a-zA-Z ]+)', expand=False).str.strip()]\n",
    "# df_test3.reset_index(drop=True,inplace=True)\n",
    "# df_test3['Recommendation'].str.extract('([a-zA-Z ]+)', expand=False).str.strip()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "##Splitting sentences on full stop and cleaning\n",
    "\n",
    "s=df_test2[\"Recommendation\"].str.split('.').apply(pd.Series).stack()\n",
    "s.index = s.index.droplevel(-1) # to line up with df's index\n",
    "s.name='Recommendation' # it gives name to series ... for joining as column below. After joining with dataframe- my_df, \n",
    "# #                             this will become the name of the column\n",
    "s.replace('',np.nan,inplace=True)\n",
    "s.dropna(inplace=True)\n",
    "\n",
    "del df_test2['Recommendation']\n",
    "my_df=df_test2.join(s)\n",
    "my_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "my_df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# a=my_df.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# aa=a[~a.Recommendation.str.contains(r'\\d')]\n",
    "# aa"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# my_df_null=my_df[my_df['Recommendation'].isnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "All_data=pd.concat([my_df,df_test2_opposite,nulls],axis=0) # appending non null '.' splitted data and null values dataframe into one"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "All_data1=All_data.reset_index()\n",
    "print(All_data1.dtypes)\n",
    "All_data1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "All_data1['Recommendation']= All_data1['Recommendation'].astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# All_data2=All_data1.replace({'Recommendation': {'-': '', '&': 'and', 'Na':'Nothing',pd.np.nan: 'None' }}, regex=True)\n",
    "# All_data2.to_csv('C:\\\\Users\\\\MalikM\\\\Documents\\\\exit interview\\\\b.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import nltk as nltk\n",
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "count=0\n",
    "aspect=[]\n",
    "for i in All_data1.index:\n",
    "# while i <86:\n",
    "# for i in range(86):\n",
    "# for i in range(85):\n",
    "    \n",
    "#     dd=pd.DataFrame()\n",
    "#     count=count+1\n",
    "    new_review=All_data1['Recommendation'][i]\n",
    "#     print(new_review)\n",
    "\n",
    "#     new_review='I tried to share my views with higher authorities for disputes with team leader but nothing changed, infect my team leader came to me and said that I tried with higher authority for disputes with him but nothing will change.. This is how a agent level person is facing issues while working in Vodafone, so please try to change this for betterment of lower level employees. Thank you.'\n",
    "    chunks = [(chunk.root.text) for chunk in nlp(new_review).noun_chunks if chunk.root.pos_ == 'NOUN']\n",
    "    new_review_aspect_terms = ' '.join(chunks)\n",
    "    new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])\n",
    "\n",
    "    new_review_category = label_encoder.inverse_transform(aspect_categories_model.predict_classes(new_review_aspect_tokenized))\n",
    "#     print(count)\n",
    "#     print(new_review_category)\n",
    "    aspect.append(new_review_category)\n",
    "    \n",
    "     #df_test4['aspect'][i]=pd.Series(new_review_category)\n",
    "\n",
    "\n",
    "    \n",
    "All_data1['aspect']=pd.Series(aspect)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "All_data1.to_csv('C:\\\\Users\\\\MalikM\\\\Documents\\\\exit interview\\\\DEC\\\\demo1_output.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### till here - below tried codes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# from textblob import TextBlob"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# def get_textBlob_score(All_data2['Recommendation']):\n",
    "#     # This polarity score is between -1 to 1\n",
    "#     polarity = TextBlob(sent).sentiment.polarity\n",
    "#     return polarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# sent=\"i am not a very good boy\"\n",
    "# get_textBlob_score(sent)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df_test3=pd.DataFrame(['Recommendation'].str.extract('([a-zA-Z ]+)', expand=False).str.strip())\n",
    "# df_test3.dtypes\n",
    "# type(df_test3)\n",
    "# # len(df_test3)\n",
    "# # df_test3.head()\n",
    "# # print(df_test3.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import nltk as nltk\n",
    "# import pandas as pd\n",
    "# import numpy as np\n"
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
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# len(df_test3)\n",
    "# df_test4=df_test3.reset_index(drop=True)\n",
    "# # df_test4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# count=0\n",
    "# for i in df_test4.index:\n",
    "#     aspect=[]\n",
    "#     dd=pd.DataFrame()\n",
    "#     count=count+1\n",
    "#     new_review=df_test4['Recommendation'][i]\n",
    "# #     print(new_review)\n",
    "# count\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# All_data1.to_csv('C:\\\\Users\\\\MalikM\\\\Documents\\\\exit interview\\\\aaa.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# count=0\n",
    "# aspect=[]\n",
    "# for i in All_data2.index:\n",
    "# # while i <86:\n",
    "# # for i in range(86):\n",
    "# # for i in range(85):\n",
    "    \n",
    "# #     dd=pd.DataFrame()\n",
    "# #     count=count+1\n",
    "#     new_review=All_data2['Recommendation'][i]\n",
    "# #     print(new_review)\n",
    "\n",
    "# #     new_review='I tried to share my views with higher authorities for disputes with team leader but nothing changed, infect my team leader came to me and said that I tried with higher authority for disputes with him but nothing will change.. This is how a agent level person is facing issues while working in Vodafone, so please try to change this for betterment of lower level employees. Thank you.'\n",
    "#     chunks = [(chunk.root.text) for chunk in nlp(new_review).noun_chunks if chunk.root.pos_ == 'NOUN']\n",
    "#     new_review_aspect_terms = ' '.join(chunks)\n",
    "#     new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])\n",
    "\n",
    "#     new_review_category = label_encoder.inverse_transform(aspect_categories_model.predict_classes(new_review_aspect_tokenized))\n",
    "# #     print(count)\n",
    "# #     print(new_review_category)\n",
    "#     aspect.append(new_review_category)\n",
    "    \n",
    "# # print(aspect)\n",
    "#     #df_test4['aspect'][i]=pd.Series(new_review_category)\n",
    "# #    print(aspect)\n",
    "\n",
    "# print(type(aspect))\n",
    "\n",
    "    \n",
    "# All_data2['aspect']=pd.Series(aspect)\n",
    "# print(All_data2)\n",
    "# #     print(dd)\n",
    "# #     print(new_review_category)\n",
    "#     #     print(new_review_category)\n",
    "# # print(count)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# #print(df_test4['aspect'])\n",
    "# print(type(new_review_category))\n",
    "\n",
    "# a=[1,2,3,4]\n",
    "# for i in a:\n",
    "#     print(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# for i in range(86):\n",
    "#     print (i)"
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
    "################################################################################################################\n",
    "# --- freq words + regex chunking + filtering -- input for topic modelling -----"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import spacy\n",
    "#nlp = spacy.load('en', disable=['parser', 'ner'])\n",
    "#doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')\n",
    "nlp=spacy.load('en_core_web_sm')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "from textblob import TextBlob\n",
    "\n",
    "import collections"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df4_1=pd.read_csv(\"C:\\\\Users\\\\MalikM\\\\Documents\\\\CVM TEAM -PUNE\\\\UK\\\\BB Combined\\\\SOHO Broadband May_June.csv\",encoding='ISO-8859-1')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2=pd.DataFrame(df4_1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2[\"review\"] = df_2['Verbatim'].str.lower()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2['Themes '].unique()\n",
    "\n",
    "df_2['Themes '].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2['Themes '][df_2['Themes ']=='Sales/Upgrade']='Sales / Upgrade' # replacing Sales/Upgrade with Sales / Upgrade\n",
    "\n",
    "df_2['Themes '].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "# train=df_2[~((df_2['Themes ']=='Uncategorised') | (df_2['Themes ']=='COVID-19') | (df_2['Themes '].isnull()))]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "\n",
    "\n",
    "# test=df_2[(df_2['Themes ']=='Uncategorised') | (df_2['Themes ']=='COVID-19') | (df_2['Themes '].isnull())]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df_2_=df_2_[~df_2_['review'].isin(['1','2','3','4','5','6','7','8','9','-',''])]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df_2[['Verbatim','review']].head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_2.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# df_3=df_2['review'].dropna()\n",
    "df_3=df_2_[~(df_2_['review'].isnull())]\n",
    "df_3.reset_index(drop=True,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df4_1=pd.read_csv(\"C:\\\\Users\\\\MalikM\\\\Documents\\\\CVM TEAM -PUNE\\\\UK\\\\CARE COMBINED\\\\SOHO Care detractors JUN_MAY_combined.csv\",encoding='ISO-8859-1')\n",
    "# df_2=pd.DataFrame(df4_1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# taking out noun terms\n",
    "# df_2[\"review\"] = df_2['Verbatim'].str.lower()\n",
    "# df_2['Themes '].unique()\n",
    "# df_2['Themes '][df_2['Themes ']=='Sales/Upgrade']='Sales / Upgrade' # replacing Sales/Upgrade with Sales / Upgrade\n",
    "# # df_3=df_2['review'].dropna()\n",
    "# df_3=df_2_[~(df_2_['review'].isnull())]\n",
    "# df_3.reset_index(drop=True,inplace=True)\n",
    "\n",
    "aspect_terms = []\n",
    "for review in nlp.pipe(df_3.review):\n",
    "    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN'] \n",
    "    aspect_terms.append(' '.join(chunks))\n",
    "    \n",
    "\n",
    "df_3['aspect_terms'] = aspect_terms\n",
    "# df_3.head(10)\n",
    "\n",
    "df_3['polarity']=df_3.apply(lambda x : TextBlob(x['review']).sentiment.polarity, axis=1)\n",
    "df_3['subjectivity']=df_3.apply(lambda x : TextBlob(x['review']).sentiment.subjectivity,axis=1)\n",
    "\n",
    "df_3['sentiment']=np.where(df_3['polarity']==0,'Neutral',np.where(df_3['polarity']>0,'Positive','Negative'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_3.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df_cust_service_neg=df_3[(df_3['Themes ']=='Customer Service') & (df_3['sentiment'].isin(['Positive']))]\n",
    "# df_cust_service_neg=df_3[(df_3['Themes ']=='Billing') ]\n",
    "df_cust_service_neg=df_3[(df_3['Themes ']=='Customer Service') ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_cust_service_neg.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lis_df_cust_service_neg=' '.join(df_cust_service_neg['review'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lis_df_cust_service_neg"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from nltk.corpus import stopwords \n",
    "from nltk.tokenize import word_tokenize \n",
    "  \n",
    "# example_sent = \"This is a sample sentence, showing off the stop words filtration.\"\n",
    "  \n",
    "stop_words = set(stopwords.words('english')) \n",
    "  \n",
    "word_tokens = word_tokenize(lis_df_cust_service_neg) \n",
    "  \n",
    "filtered_sentence = [w for w in word_tokens if not w in stop_words] \n",
    "  \n",
    "# filtered_sentence = [] \n",
    "  \n",
    "# for w in word_tokens: \n",
    "#     if w not in stop_words: \n",
    "#         filtered_sentence.append(w) \n",
    "  \n",
    "# print(word_tokens) \n",
    "print(filtered_sentence) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# lis_df_cust_service_neg_=lis_df_cust_service_neg.split()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lis_df_cust_service_neg_counts = collections.Counter(filtered_sentence)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lis_df_cust_service_neg_counts.most_common(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## regex and pos and chunking part---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# import nltk\n",
    "# tokens = nltk.word_tokenize(lis_df_cust_service_neg)\n",
    "# # print(tokens)\n",
    "# tag = nltk.pos_tag(tokens)\n",
    "# grammar = r\"\"\"\n",
    "#   NP: {<NN.?>*<VBD.?>*<RB.?>*<VB.?>*<JJ.?>*<NN.?>*<NN.?>?}\n",
    "#   }<MD>{\"\"\"\n",
    "# cp  =nltk.RegexpParser(grammar)\n",
    "# result = cp.parse(tag)\n",
    "# print(result)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# result_=pd.Series(result)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# result_df=pd.DataFrame({'res':result_})\n",
    "# result_df.columns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# result_df=result_df.replace(['\\)','\\(','\\'',',','/','\\n','.','NN','VBD','RB','JJ','CC','NP','IN','DT','VBN','VBG','MD','VB','PRP$','S'],'',regex=True).astype(str)\n",
    "# result_df['totalwords'] = result_df['res'].str.split().str.len()\n",
    "# # result_df['totalwords'] = df['col'].str.count(' ') + 1\n",
    "\n",
    "# result_df=result_df.sort_values('totalwords', ascending=False)\n",
    "# # result_df_final=pd.DaraFrame(result_df.res)\n",
    "# result_df=result_df.replace(['\\)','\\(','\\'',',','/','\\n','Z','N','P','NN','VBD','RB','JJ','CC','NP','IN','DT','VBN','VBG','MD','VB','PRP$','S'],'',regex=True).astype(str)\n",
    "# result_df.totalwords.astype(int)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# result_df.dtypes\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# general=['vodafone','leave','stay','loyality','language','barrier','service','customer','option','satisfy','english','satisfaction','charge','cheap','cost','loss','switch','move','change','end','contract','continue','explain','understand','fluency','speak','overcharge','delay']\n",
    "general=['vodafone','leave','stay','loyality','option','satisfy','satisfaction','loss','switch','move','change','end','continue']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# li_20=pd.Series(lis_df_cust_service_neg_counts.most_common(20))\n",
    "li_20=pd.DataFrame(lis_df_cust_service_neg_counts.most_common(35), columns=['words','counts'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "li_20_cleaned=li_20[(~(li_20['words'].isin(['.',',','!','?',\"n't\",\"'s\",'...'])))] ['words']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "li_20_cleaned_li=li_20_cleaned.tolist() # list of most frequesnt words\n",
    "## combine these with general"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# mylist=li_20_cleaned_li + general\n",
    "mylist=list(filter(lambda x : x not in general, li_20_cleaned))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "mylist"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# mylist=['broadband','problem','internet'] # from john - most frequent words used for broadband themes\n",
    "# mylist=['billing','bill'] # from john - most frequent words used for broadband themes\n",
    "# mylist=['call','time','call','service','vodafone','broadband','wait','resolve','customer','bussiness'] # from john - most frequent words used for broadband themes\n",
    "\n",
    "# mylist=['good','better','nice','very good']\n",
    "pattern = '|'.join(mylist)\n",
    "# mylist=['vodafone','leave','stay','loyality','language','barrier','service','customer','option','satisfy','satisfaction','charge','cheap','cost','loss','agent','change','leave','move','call','phone','resolv','time','vodafon','back','issu','problem','servic','still','get','custom','the','account','understand','sort','bill','didnt','hour','tri','promis','month','will','one','help','told','wait','advisor','cut','just','spoke']\n",
    "# ['vodafone','leave','stay','loyality','language','barrier','call',\t'phone',\t'resolv',\t'time',\t'vodafon',\t'back',\t'issu',\t'problem',\t'servic',\t'still',\t'get',\t'custom',\t'the',\t'account',\t'understand','sort','bill','didnt','hour','tri','promis','month','will','one','help','told','wait','advisor','cut','just','spoke']\n",
    "# 'call',\t'phone',\t'resolv',\t'time',\t'vodafon',\t'back',\t'issu',\t'problem',\t'servic',\t'still',\t'get',\t'custom',\t'the',\t'account',\t'understand',\t'sort',\t'bill',\t'didnt',\t'hour',\t'tri',\t'promis',\t'month',\t'will',\t'one',\t'help',\t'told',\t'wait',\t'advisor',\t'cut',\t'just',\t'spoke',"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pattern"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_cust_service_neg.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_cust_service_neg.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# these most frequent words are the searched in the Regex-chunked broadband comments.\n",
    "# result_df_billing=result_df[result_df['res'].str.contains(\"bill\") & (result_df['totalwords']>2)]\n",
    "# result_df_broadband=result_df[result_df['res'].str.contains(pattern) & (result_df['totalwords']>1)]\n",
    "# result_df_billing=result_df[result_df['res'].str.contains(pattern) &(result_df.totalwords.astype(int)> 1)]\n",
    "\n",
    "result_df_CustService=df_cust_service_neg[df_cust_service_neg['review'].str.contains(pattern) ]\n",
    "# result_df_CustService=result_df[result_df['res'].str.contains(pattern) &(result_df.totalwords.astype(int)> 2)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "result_df_CustService.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "result_df_CustService.to_csv(\"C:\\\\Users\\\\MalikM\\\\Documents\\\\CVM TEAM -PUNE\\\\UK\\\\BB Combined\\\\input_for_topic_modelling_Cust_service.csv\")"
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
   "source": []
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
   "source": []
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
