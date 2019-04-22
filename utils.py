# -*- coding:utf-8 -*-
from __future__ import print_function
from datetime import datetime
import numpy as np
import json
import sys 
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import TreebankWordTokenizer
import string
from datetime import timedelta
import time
# print log on screen and save log in file
CATEGORIE_ID = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
def print_log(*args,**kwargs):
	print(*args)
	if len(kwargs) > 0:
		print(*args,**kwargs)
	return None

def time_cost(start_time):
	end_time = time.time()
	diff = end_time - start_time
	return timedelta(seconds=(int(round(diff))))

def extract_embedding(embedding_path):
	embedding = {}
	with open(embedding_path,'r') as fr:
		for line in fr:
			# line = line.strip()
			temp = line.strip().split()
			# print(temp)
			if len(temp) !=301:
				continue
			embedding[temp[0].strip()] = np.array(temp[1:],dtype=np.float32)
	# json.dump(embedding,open('./DATA/extracted_embedding.json','w'))
	return embedding

def load_embedding(embedding_dict,voc):
	embedding = np.random.normal(loc = 0.0 , scale = 1.0 , size = [len(voc),300])
	count = 0
	for word,index in voc:
		if word in embedding_dict:
			embedding[index] = embedding_dict.get(word)
			count+=1
	# print(embedding,count)
	print(count)
	np.save('./DATA/embedding.npy',embedding)

def norm_embedding(embedding):
	norms = np.linalg.norm(embedding,axis=1).reshape((-1,1))
	return embedding/norms	

def preprocess_data(json_path,txt_path):
	f = open(txt_path,'w')
	with open(json_path) as js:
		for line in js :
			line = json.loads(line.strip())
			if line['gold_label'] != '-':
				f.write('|||'.join([ line['gold_label'],line['sentence1'],line['sentence2'] ]))
				f.write('\n')
	f.close()

def tokenize_data(data_path,save_path):
	# pattern = re.compile(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*")
	# fs = open(save_path,'w')
	voc = set()
	words = []
	with open(data_path,'r') as f:
		for line in f.readlines():
			temp = line.strip().split('|||')
			for i in range(1,3):
				words.extend(TreebankWordTokenizer().tokenize(temp[i].lower().strip()))

	for word in words:
		if word.isalpha() == True :
			voc.add(word.strip())

	with open(save_path,'w') as fw:
		for word in voc :
			fw.write(word)
			fw.write('\n')

def tokenize_file(data_path,save_path):
	# input label|||sentence1|||sentence2 
	# ouput label|||sentence1.tok|||sentence2.tok
	fw = open(save_path,'w')
	with open(data_path,'r') as fr:
		for line in fr:
			temp = line.strip().split('|||')
			# print(temp[0],'\n',temp[1],'\n',temp[2])
			sentence1 = TreebankWordTokenizer().tokenize(temp[1].lower().strip())
			sentence2 = TreebankWordTokenizer().tokenize(temp[2].lower().strip())
			# print(temp[0],'\n',sentence1,'\n',sentence2)
			sentence1_tok = [word.strip() for word in sentence1 if word not in string.punctuation]
			sentence2_tok = [word.strip() for word in sentence2 if word not in string.punctuation]
			# print(temp[0],'\n',temp[1],'\n',sentence1_tok,'\n',temp[2],'\n',sentence2_tok)
			line = '|||'.join([temp[0].strip(),' '.join(sentence1_tok),' '.join(sentence2_tok)])
			# print(line)
			fw.write(line+'\n')
	fw.close()

def build_vocab(data_path,save_path):
	voc = {}
	voc['<<UNK>>'] = 1
	voc['<<PAD>>'] = 0
	with open(data_path,'r') as fr:
		for line in fr:
			words = []
			temp = line.strip().split('|||')
			words.extend(temp[1].strip().split())
			words.extend(temp[2].strip().split())
			for word in words:
				if word not in voc:
					voc[word] = len(voc)
	voc = sorted(voc.items(),key = lambda x:x[1])
	json.dump(voc,open(save_path,'w'))

def input_process(line,voc,maxlen=100):
	words = TreebankWordTokenizer().tokenize(line.lower().strip())
	words = [word.strip() for word in words	if word.strip().isalpha()]
	sequence_input = np.zeros(maxlen,dtype=np.int32)
	for index,word in enumerate(words):
		if index == maxlen-1 :
			break
		elif  word.isalpha() == True:
			sequence_input[index] = voc.get(word,1)
	return sequence_input.astype(dtype=np.int32),len(words)

def sentence2index(line,voc,maxlen=100):
	words = line.strip().lower().split()
	input_index = np.zeros(maxlen,dtype=np.int32)
	for index,word in enumerate(words):
		input_index[index] = voc.get(word,1)
	# print(words)
	# print(input_index)
	return input_index.astype(np.int32),len(words)

def generate_file(path,voc):
	sentence1 , sentence2 , length1 , length2 , labels = [] , [] , [] , [] , []
	with open(path,'r') as f:
		for line in f:
			temp = [x.strip() for x in line.lower().strip().split('|||')]

			labels.append(CATEGORIE_ID[temp[0]])
			input1,len1 = sentence2index(temp[1],voc)
			input2,len2 = sentence2index(temp[2],voc)
			sentence1.append(input1)
			sentence2.append(input2)
			length1.append(len1)
			length2.append(len2)


	return np.array(sentence1,dtype=np.int32),np.array(sentence2,dtype=np.int32) \
				,np.array(length1,dtype=np.int32),np.array(length2,dtype=np.int32),np.array(labels,dtype=np.int32)

def generate_batch(input_sentence1,input_sentence2,length1,length2,label,batch_size = 4 ,shuffle_flag = True):
	if shuffle_flag:
		permutation = list(np.random.permutation(len(label)))
		input_sentence1 = input_sentence1[permutation,]
		input_sentence2 = input_sentence2[permutation,]
		length1 = length1[permutation]
		length2 = length2[permutation]
		label = label[permutation]

	batch_nums = (len(label)-1) // batch_size + 1
	for i in range(batch_nums):
		start = i*batch_size
		end = min((i+1)*batch_size,len(label))
		yield np.array(input_sentence1[start:end],dtype=np.int32) , np.array(input_sentence2[start:end],dtype=np.int32) , \
		np.array(length1[start:end],dtype=np.int32) , np.array(length2[start:end],dtype=np.int32) , \
		np.array(label[start:end],dtype=np.int32)

def config_save(args,others=None):
	path = './config.txt'
	f = open(path,'a')
	msg = '-'*20 + ' Config Imformation '+ '-'*20
	f.write(msg+'\n')
	print(msg)
	for key,value in vars(args).items():
		msg = '%-15s : %s' %(key,value)
		print(msg)
		f.write(msg+'\n')
	if others!=None:
		for key,value in others.items():
			msg = '%-15s : %s' %(key,value)
			print(msg)
			f.write(msg+'\n')
	msg = '-'*60
	print(msg)
	f.write(msg+'\n')
	f.close()

def step1():
	preprocess_data('./snli_1.0/snli_1.0_dev.jsonl','./DATA/dev.txt')
	preprocess_data('./snli_1.0/snli_1.0_train.jsonl','./DATA/train.txt')
	preprocess_data('./snli_1.0/snli_1.0_test.jsonl','./DATA/test.txt')
	pass

def step2():
	# tokenize data
	# return label|||sentence1_tok|||sentence2_tok
	dev_path = './DATA/dev.txt'
	test_path = './DATA/test.txt'
	train_path = './DATA/train.txt'

	tokenize_file(dev_path,'./DATA/dev_tok.txt')
	tokenize_file(test_path,'./DATA/test_tok.txt')
	tokenize_file(train_path,'./DATA/train_tok.txt')
	print('Finish file tokenize')

def step3():
	# build vocabulary
	build_vocab('./DATA/train_tok.txt','./DATA/vocabulary.json')
	# voc = json.load(open('./DATA/vocabulary.json','r'))
	# print(voc)
	# for word,index in voc:
		# print(word,'\t',index)
	# voc = dict((temp[0],temp[1]) for temp in voc)
	# print('\n',voc)

def step8():
	voc = json.load(open('./DATA/vocabulary.json','r'))
	voc = dict((temp[0],temp[1]) for temp in voc)
	with open('./DATA/dev_tok.txt','r') as f:
		for line in f:
			temp = line.lower().strip().split('|||')
			sentence2index(temp[1],voc)
			sentence2index(temp[2],voc)

def step9():
	#generate file and turn it into index
	voc = json.load(open('./DATA/vocabulary.json','r'))
	voc = dict((temp[0],temp[1]) for temp in voc)
	data = generate_file('./DATA/dev_tok.txt',voc)
	# for i in range(len(data)) :
		# print(data[i].shape)
		# print(data[i])

def step4(embedding_path):
	## extracted embedding and save it 
	voc = json.load(open('./DATA/vocabulary.json','r'))
	# voc = dict((temp[0],temp[1]) for temp in voc)
	print('extract data from glove')

	embedding_dict = extract_embedding('/data00/home/labspeech_intern/leixiaojun/my_project/text_matching_2ND/pretain_embedding/glove.840B.300d.txt')
	load_embedding(embedding_dict,voc)

def check(data):
	for i in range(10):
		print(data[i])
	pass

if __name__ == '__main__':
	# embedding_path = './my_project/text_matching_2ND/pretain_embedding/glove.840B.300d.txt'
	step1()
	step2()
	step3()
	step4(None)	
	pass
