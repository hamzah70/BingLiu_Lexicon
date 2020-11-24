#generic modules
import os.path
import pandas as pd
import numpy as np
import pickle

#stanza
import stanza

#nltk
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

#genshim
from gensim.models import Word2Vec
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# wv = api.load('word2vec-google-news-300')
# print(wv['king'])

#function for preprocessing english txt. replacing each word with its lemmatized version -> stemmed version.
def preprocessEnglish(data):
	lemmatizer = WordNetLemmatizer()
	ps = PorterStemmer()
	for i, line in enumerate(data):
		for j, word in enumerate(line):
			word = lemmatizer.lemmatize(word)
			data[i][j] = ps.stem(word) # stemming
	f = open('preprocessed/english.txt', 'w')# preprocessed files are dumped onto preprocessed folder
	for line in data:
		line.append('\n')
		line = ' '.join(line)
		f.write(line)
	f.close()

#function for preprocessing hindi txt. replacing each word with its lemmatized version.
def preprocessHindi(data):
	nlp = stanza.Pipeline('hi', use_gpu=True)
	for i, line in enumerate(data):
		doc = nlp(line)
		result = ''
		for sentence in doc.sentences:
			for j, word in enumerate(sentence.words):
				result += word.lemma + ' ' #lemmatizing
		data[i] = result[:-1]
	f = open('preprocessed/hindi.txt', 'w') #preprocessed files are dumped onto preprocessed folder
	for line in data:
		line += '\n'
		f.write(line)
	f.close()


def createL1(bingLiuDict, engHindiDict): #creating L1.csv which contains all possible english to hindi word mapping with its polarity
	arr = []
	for engWord, polarity in bingLiuDict.items():
		if engWord in engHindiDict:
			for hindiWord in engHindiDict[engWord]:
				if engWord != hindiWord:
					d = [engWord, hindiWord, polarity] #all exsting pairs are added to L1
					# print(d)
					arr.append(d)
	df = pd.DataFrame(arr)
	df.to_csv("L1.csv", index=False, header=False) #L1.csv saved





def gloveTrain(data, filename): #function to train glove
	filename = "models/glove/" + filename #glove vector representation.
	f = open(filename + ".txt", 'r')
	data = f.read().split('\n')[:-1]
	vectorDict = {}
	l = len(data[0].split(' ')) - 1
	for d in data:
		d = d.split(' ')
		arr = np.zeros((l,))
		for i, num in enumerate(d[1:]):
			arr[i] = float(num)
		vectorDict[d[0]] = arr
	f = open(filename + ".pkl", 'wb') #dumping list of vectors
	pickle.dump(vectorDict, f)


def closest(bingLiuDict, engHindiDict, wvEnglish, wvHindi):
	l1df = pd.read_csv('L1.csv')
	newAdditions = []
	all_pairs = {}
	total_pairs = {}
	
	#initialization
	for i in range(len(l1df)):
		englishWord = l1df.iloc[i,0]
		hindiWord = l1df.iloc[i,1]
		polarity = bingLiuDict[englishWord]
		all_pairs[(englishWord,hindiWord,polarity)] = 1 
		total_pairs[(englishWord,hindiWord,polarity)] = 1

	while(len(all_pairs) != 0): #loops untill no new additions are seen.
		temp = {}
		for i in all_pairs:
			englishWord = i[0]
			hindiWord = i[1]
			polarity = i[2]

			try:
				englishSimilar = wvEnglish.most_similar(positive=[englishWord], topn=5) #checks if word is in vocabulary or not
			except KeyError:
				continue

			try:
				hindiSimilar = wvHindi.most_similar(positive=[hindiWord], topn=5)#checks if word is in vocabulary or not
			except KeyError:
				continue

			for ewt in englishSimilar:
				ew = ewt[0]
				if ew in engHindiDict:
					for hwt in hindiSimilar:
						hw = hwt[0]
						if hw in engHindiDict[ew]:
							arr = (ew, hw, polarity)
							if(arr not in total_pairs): #checks if the following pair is already added or not
								newAdditions.append(arr)
								temp[arr] = 1
								total_pairs[arr] = 1
		all_pairs = temp; #new additions stored
		# print(all_pairs)
		# print(len(all_pairs))
	return newAdditions


def findTopClosestWord2Vec(bingLiuDict, engHindiDict):
	word2vecEnglish = Word2Vec.load("models/word2vec/english.model")
	word2vecHindi = Word2Vec.load("models/word2vec/hindi.model")

	wvEnglish = word2vecEnglish.wv #keyed vectors for english
	wvHindi = word2vecHindi.wv #keyed vectors for hindi

	return closest(bingLiuDict, engHindiDict, wvEnglish, wvHindi)


def findTopClosestGlove(bingLiuDict, engHindiDict):
	gloveEnglish_file = 'models/glove/english.txt'
	tmp_file_english = 'testEnglish.txt'

	gloveHindi_file = 'models/glove/hindi.txt'
	tmp_file_hindi = 'testHindi.txt'

	_ = glove2word2vec(gloveEnglish_file, tmp_file_english)
	modelEnglish = KeyedVectors.load_word2vec_format(tmp_file_english)

	_ = glove2word2vec(gloveHindi_file, tmp_file_hindi)
	modelHindi = KeyedVectors.load_word2vec_format(tmp_file_hindi)

	return closest(bingLiuDict, engHindiDict, modelEnglish, modelHindi) #closest new words are added.


def word2VecTrain(data, filename,window1):
	model = Word2Vec(sentences=data, window=window1, min_count=1, workers=4,seed = 1)
	filename = "models/word2vec/" + filename
	model.save(filename) #trained model is saved onto disk
	# print(model.wv["ब्रॉडकास्टर"])
	# print(model.wv["कॉन्टैक्ट"])
	# print(model.wv["नहीं"])
	# print(model.wv["उपलब्ध"])

if __name__ == "__main__":
	f = open("assignment_4_files/BingLiu.csv", "r")
	data = f.read().split("\n")[:-1]
	bingLiuDict = {}
	for d in data:
		d = d.split("\t")
		bingLiuDict[d[0]] = d[1] #bing liu dictionaryc construction.

	f = open("assignment_4_files/english-hindi-dictionary.txt", 'rb')
	data = f.read().split(b"\n")[:-1]
	engHindiDict = {}
	for d in data: # iterating over all english hindi pairs.
		d = d.split(b" ||| ")
		d[0] = d[0].decode('utf-8')
		d[1] = d[1].decode('utf-8')
		if d[0] not in engHindiDict:
			# engHindiDict[d[0]] = [d[1]]
			engHindiDict[d[0]] = {d[1]:1}
		else:
			# engHindiDict[d[0]].append(d[1])
			engHindiDict[d[0]][d[1]] = 1


	createL1(bingLiuDict, engHindiDict) #creates and saves L1.csv

	f1 = open("assignment_4_files/english.txt", "r")
	f2 = open("assignment_4_files/hindi.txt", "rb")

	data1 = f1.read().split("\n")[:-1]
	data2 = f2.read().split(b"\n")[:-1]

	for i, line in enumerate(data1):
		data1[i] = line.split(' ')

	rawHindi = []
	for i, line in enumerate(data2):
		data2[i] = line.decode('utf-8').strip()
		rawHindi.append(data2[i])
		data2[i] = data2[i].split(' ')

	# preprocessEnglish(data1)
	# preprocessHindi(rawHindi)

	
	if not os.path.isfile("models/word2vec/english.model"):
		word2VecTrain(data1, "english.model",39) 
	if not os.path.isfile("models/word2vec/hindi.model"):
		word2VecTrain(data2, "hindi.model",39)

	if not os.path.isfile("models/glove/english.pkl"):
		gloveTrain(data1, "english")
	if not os.path.isfile("models/glove/hindi.pkl"):
		gloveTrain(data2, "hindi")


	#new additions are found.
	l1 = pd.read_csv("L1.csv").values.tolist()
	print("top closest word2Vec")
	word2vecAddition = findTopClosestWord2Vec(bingLiuDict, engHindiDict)
	print(len(set(word2vecAddition)))
	word2vecAddition = list(set(word2vecAddition))
	print(word2vecAddition)
	w2vL1 = l1 + word2vecAddition
	w2vdf = pd.DataFrame(w2vL1)
	w2vdf.to_csv("L1_word2vec.csv", index=False, header=False)

	# print(word2vecAddition)
	print("top closest glove")
	gloveAddition = findTopClosestGlove(bingLiuDict, engHindiDict)
	gloveAddition = list(set(gloveAddition))
	print(len(gloveAddition))
	print(gloveAddition)
	gloveL1 = l1 + gloveAddition
	glovedf = pd.DataFrame(gloveL1)
	glovedf.to_csv("L1_glove.csv", index=False, header=False)

	l1 = l1 + word2vecAddition + gloveAddition
	df = pd.DataFrame(l1)
	df.to_csv("L1_total.csv", index=False, header=False)

	print("Total unique additions")
	print(len(word2vecAddition + gloveAddition))
	print(word2vecAddition + gloveAddition)