from model import Decompose_Model
import utils
import train
import json
import numpy as np
import tensorflow as tf
CATEGORIE_ID = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

def generate_batch(input_sentence1,input_sentence2,length1,length2,label,batch_size = 4 ):	
	batch_nums = (len(label)-1) // batch_size + 1
	for i in range(batch_nums):
		start = i*batch_size
		end = min((i+1)*batch_size,len(label))
		yield np.array(input_sentence1[start:end],dtype=np.int32) , np.array(input_sentence2[start:end],dtype=np.int32) , \
		np.array(length1[start:end],dtype=np.int32) , np.array(length2[start:end],dtype=np.int32) , \
		np.array(label[start:end],dtype=np.int32) , np.arange(start,end)

def generate_file(path,voc):
	sentence1 , sentence2 , length1 , length2 , labels , original_input1 , original_input2 = [] , [] , [] , [] , [] , [] , []
	with open(path,'r') as f:
		for line in f:
			temp = [x.strip() for x in line.lower().strip().split('|||')]

			labels.append(CATEGORIE_ID[temp[0]])
			input1,len1 = utils.sentence2index(temp[1],voc)
			input2,len2 = utils.sentence2index(temp[2],voc)
			sentence1.append(input1)
			sentence2.append(input2)
			original_input1.append(temp[1])
			original_input2.append(temp[2])
			length1.append(len1)
			length2.append(len2)


	return np.array(sentence1,dtype=np.int32),np.array(sentence2,dtype=np.int32) \
				,np.array(length1,dtype=np.int32),np.array(length2,dtype=np.int32),np.array(labels,dtype=np.int32) , \
				original_input1,original_input2
def test():
	test_path = './DATA/test_tok.txt'
	args = train.parser_setting()
	embedding = np.load(args.embedding_path).astype(np.float32)
	embedding = utils.norm_embedding(embedding)

	voc = json.load(open('./DATA/vocabulary.json','r'))
	voc = dict((temp[0],temp[1]) for temp in voc)

	test_set = utils.generate_file(test_path,voc)

	parameters = train.generate_parameters(args,voc)

	model = Decompose_Model(parameters)

	sess = tf.Session(config=train.tf_config())

	saver = tf.train.Saver()

	saver.restore(sess,args.load_model)

	test_result = train.evaluate(sess,model,test_set)

	print 'Model Performance on Test Set : %s'%test_result

def check_error():
	data_path = './DATA/test_tok.txt'
	args = train.parser_setting()
	embedding = np.load(args.embedding_path).astype(np.float32)
	embedding = utils.norm_embedding(embedding)

	voc = json.load(open('./DATA/vocabulary.json','r'))
	voc = dict((temp[0],temp[1]) for temp in voc)

	data_set = generate_file(data_path,voc)
	parameters = train.generate_parameters(args,voc)

	model = Decompose_Model(parameters)

	sess = tf.Session(config=train.tf_config())
	saver = tf.train.Saver()

	saver.restore(sess,args.load_model)

	error_result = figure_out(sess,model,data_set)
        
        '''
	for error in error_result : 
		error_index = error[0]
		error_pred = error[1]
		print '%s||%s\t%d\t%d'%(data_set[5][error_index],data_set[6][error_index],error_pred,data_set[4][error_index])
        '''


def figure_out(sess,model,data):
	batch_data = generate_batch(data[0],data[1],data[2],data[3],data[4],batch_size = 128 )
	batch_result = []
	error_index = []
	for input1,input2,length1,length2,label,original in batch_data:
		feed_dict = model.feed_dict(input1,input2,length1,length2,label, \
			None,1.0)
		batch_pred = sess.run(model.pred,feed_dict = feed_dict)
		for index,pred in enumerate(batch_pred):
			if pred != label[index] :
				error_index.append((original[index],pred))
	return error_index

if __name__ == '__main__':
    test()
    #check_error()
