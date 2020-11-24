import os.path

import pandas as pd
import numpy as np
import pickle

import stanza

from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

from gensim.models import Word2Vec
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# wv = api.load('word2vec-google-news-300')
# print(wv['king'])

def preprocessEnglish(data):
	lemmatizer = WordNetLemmatizer()
	ps = PorterStemmer()
	for i, line in enumerate(data):
		for j, word in enumerate(line):
			word = lemmatizer.lemmatize(word)
			data[i][j] = ps.stem(word)
	f = open('preprocessed/english.txt', 'w')
	for line in data:
		line.append('\n')
		line = ' '.join(line)
		f.write(line)
	f.close()

def preprocessHindi(data):
	nlp = stanza.Pipeline('hi', use_gpu=True)
	for i, line in enumerate(data):
		doc = nlp(line)
		result = ''
		for sentence in doc.sentences:
			for j, word in enumerate(sentence.words):
				result += word.lemma + ' '
		data[i] = result[:-1]
	f = open('preprocessed/hindi.txt', 'w')
	for line in data:
		line += '\n'
		f.write(line)
	f.close()


def createL1(bingLiuDict, engHindiDict):
	arr = []
	for engWord, polarity in bingLiuDict.items():
		if engWord in engHindiDict:
			for hindiWord in engHindiDict[engWord]:
				if engWord != hindiWord:
					d = [engWord, hindiWord, polarity]
					# print(d)
					arr.append(d)
	df = pd.DataFrame(arr)
	df.to_csv("L1.csv", index=False, header=False)





def gloveTrain(data, filename):
	filename = "models/glove/" + filename
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
	f = open(filename + ".pkl", 'wb')
	pickle.dump(vectorDict, f)


def closest(bingLiuDict, engHindiDict, wvEnglish, wvHindi):
	l1df = pd.read_csv('L1.csv')
	newAdditions = []
	for i in range(len(l1df)):
		englishWord = l1df.iloc[i,0]
		hindiWord = l1df.iloc[i,1]
		polarity = bingLiuDict[englishWord]

		try:
			englishSimilar = wvEnglish.most_similar(positive=[englishWord], topn=5)
		except KeyError:
			continue

		try:
			hindiSimilar = wvHindi.most_similar(positive=[hindiWord], topn=5)
		except KeyError:
			continue

		for ewt in englishSimilar:
			ew = ewt[0]
			if ew in engHindiDict:
				for hwt in hindiSimilar:
					hw = hwt[0]
					if hw in engHindiDict[ew]:
						arr = (ew, hw, polarity)
						newAdditions.append(arr)

	return newAdditions


def findTopClosestWord2Vec(bingLiuDict, engHindiDict):
	word2vecEnglish = Word2Vec.load("models/word2vec/english.model")
	word2vecHindi = Word2Vec.load("models/word2vec/hindi.model")

	wvEnglish = word2vecEnglish.wv
	wvHindi = word2vecHindi.wv

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

	return closest(bingLiuDict, engHindiDict, modelEnglish, modelHindi)


def word2VecTrain(data, filename,window1):
	model = Word2Vec(sentences=data, window=window1, min_count=1, workers=4,seed = 1)
	filename = "models/word2vec/" + filename
	model.save(filename)
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
		bingLiuDict[d[0]] = d[1]

	f = open("assignment_4_files/english-hindi-dictionary.txt", 'rb')
	data = f.read().split(b"\n")[:-1]
	engHindiDict = {}
	for d in data:
		d = d.split(b" ||| ")
		d[0] = d[0].decode('utf-8')
		d[1] = d[1].decode('utf-8')
		if d[0] not in engHindiDict:
			# engHindiDict[d[0]] = [d[1]]
			engHindiDict[d[0]] = {d[1]:1}
		else:
			# engHindiDict[d[0]].append(d[1])
			engHindiDict[d[0]][d[1]] = 1


	createL1(bingLiuDict, engHindiDict)

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

	print("top closest word2Vec")
	word2vecAddition = findTopClosestWord2Vec(bingLiuDict, engHindiDict)
	print(len(set(word2vecAddition)))
	print(set(word2vecAddition))
	# print(word2vecAddition)
	print("top closest glove")
	gloveAddition = findTopClosestGlove(bingLiuDict, engHindiDict)
	print(len(set(gloveAddition)))
	print(set(gloveAddition))

	print("Total unique additions")
	print(len(set(word2vecAddition + gloveAddition)))
	print(set(word2vecAddition + gloveAddition))
	






