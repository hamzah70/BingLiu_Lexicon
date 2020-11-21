import pickle
# f = open('models/glove/english.pkl', 'rb')
# a = pickle.load(f)
# print(a['the'])
# print(a['I'])
# print(a['a'])
f2 = open("assignment_4_files/hindi.txt", "rb")
data2 = f2.read().split(b"\n")[:-1]

for i, line in enumerate(data2):
	data2[i] = line.decode('utf-8').strip()
	# data2[i] = data2[i].split(' ')

import stanza
nlp = stanza.Pipeline('hi', use_gpu=True)
for i, line in enumerate(data2[:3]):
	doc = nlp(line)
	result = ''
	for sentence in doc.sentences:
		for j, word in enumerate(sentence.words):
			print(word.text, word.lemma, word.pos)
			result += word.lemma + ' '




