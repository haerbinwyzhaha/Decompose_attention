# -*- coding:utf-8 -*-
import utils
from model import Decompose_Model
import tensorflow as tf
import argparse
import time
from tqdm import tqdm
import numpy as np
import time
import json
from datetime import datetime
def tf_config():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	return config

def parser_setting():
	parser = argparse.ArgumentParser("Train Parameter")
	parser.add_argument("--hidden_size",'-hz',help='the dimension of forward layers',default=200,type=int)
	parser.add_argument('--batch_size','-batch',help='batch size of input ',default=64,type=int)
	parser.add_argument('--embedding_size','-eb',help='size of embedding layer',default=300,type=int)
	parser.add_argument('--max_length','-maxlen',help='the max len of sequence input ,if not enough , will pad 0',default=100,type=int)
	parser.add_argument('--class_size','-cs',help='the class nums of label',default=3)
	parser.add_argument('--learning_rate','-lr',help='gradients descend fast',default=0.05,type=float)
	parser.add_argument('--optimizer',help='the optimizer of gradients descence',default='adagrad')
	parser.add_argument('--keep',help='the dropout_rate of neural network',default=0.8,type=float)
	parser.add_argument('--epoch',help='the epoch nums of training',default=100,type=int)
	parser.add_argument('--clip_value',help='gradients clip ',default=100,type=int)
	parser.add_argument('--load_model',help='load_model',default='./save_model/default_version.ckpt',type=str)
	parser.add_argument('--l2',help='the weight of l2 loss',default=0,type=float)
	parser.add_argument('--save_path',help='the training model save path',default='./save_model/default_version.ckpt',type=str)
	parser.add_argument('--best_path',help='save best performance path',default='./best_model/',type=str)
	parser.add_argument('--embedding_path',help='embedding path ',default='./DATA/embedding.npy')
	parser.add_argument('--save_frequency',help='save model frequency',default=60000,type=int)
	# args = parser.parse_args()
	args = parser.parse_args()	
	return args

def generate_parameters(args,voc):
	parameters={}
	parameters['hidden_size'] = args.hidden_size
	parameters['embedding_size'] = args.embedding_size
	parameters['classes'] = args.class_size
	parameters['dropout'] = args.keep
	parameters['learning_rate'] = args.learning_rate
	parameters['vocabulary_size'] =  len(voc)
	parameters['maxlen'] = args.max_length
	parameters['clip_value'] = args.clip_value
	return parameters

def evaluate(sess,model,data):
	dev_data = utils.generate_batch(data[0],data[1],data[2],data[3],data[4],batch_size=128, \
		shuffle_flag=False)
	dev_result = []
	for input1,input2,length1,length2,label in dev_data:
		feed_dict = model.feed_dict(input1,input2,length1,length2,label, \
			None,1.0)
		dev_acc = sess.run(model.acc,feed_dict=feed_dict)
		dev_result.append(dev_acc)
	return np.mean(dev_result)

def train():
	train_path = './DATA/train_tok.txt'
	dev_path = './DATA/dev_tok.txt'
	args = parser_setting()
	embedding = np.load(args.embedding_path).astype(np.float32)
	embedding = utils.norm_embedding(embedding)
	voc = json.load(open('./DATA/vocabulary.json','r'))
	voc = dict((temp[0],temp[1]) for temp in voc)

	train_set = utils.generate_file(train_path,voc)
	dev_set = utils.generate_file(dev_path,voc)

	parameters = generate_parameters(args,voc)
	model = Decompose_Model(parameters)

	saver = tf.train.Saver()

	nowtime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
	log = open('./LOG/'+nowtime,'w')
	start_time = time.time()

	sess = tf.Session(config=tf_config())
	sess.run(tf.global_variables_initializer(),{model.embedding_: embedding})

	print 'Ready for training ',time.ctime()
	total_batch = (len(train_set[4])-1) // args.batch_size +1
	step = 0
	best_train_acc = 0.
	best_dev_acc = 0.
	best_path = args.best_path + args.save_path.split('/')[-1]
	args.best_path = best_path	

	for epoch in range(args.epoch):
		total_acc = []
		total_loss = []
		train_data = tqdm(utils.generate_batch(train_set[0],train_set[1],train_set[2],train_set[3], \
			train_set[4],batch_size=args.batch_size),total = total_batch)
		for input1,input2,length1,length2,label in train_data:
			feed_dict = model.feed_dict(input1,input2,length1,length2,label, \
				args.learning_rate,args.keep)
			_,loss_val,acc_val = sess.run([model.train,model.loss,model.acc],feed_dict=feed_dict)
			total_loss.append(loss_val)
			total_acc.append(acc_val)
			step+=1
			if step % 100 == 0:
				train_data.set_description('Epoch %s Loss %s Acc %s'%(epoch+1,loss_val,acc_val))
			if step % args.save_frequency == 0:
				saver.save(sess,args.save_path)

		if best_train_acc < np.mean(total_acc) :
			best_train_acc = np.mean(total_acc)
		utils.print_log('Epoch %s Mean Loss %s Acc %s'%(epoch+1,np.mean(total_loss),np.mean(total_acc)),file=log)
		# print('Epoch %s Mean Loss %s Acc %s'%(epoch+1,np.mean(total_loss),np.mean(total_acc)))

		dev_acc = evaluate(sess,model,dev_set)
		if dev_acc > best_dev_acc:
			best_dev_acc = dev_acc
			saver.save(sess,args.best_path)
		utils.print_log('Epoch %s Dev Acc %s'%(epoch+1,dev_acc),file=log)
		# print('Epoch %s Dev Acc %s'%(epoch+1,dev_acc))

	model_performance = {}
	model_performance['BestDev.Acc'] = best_dev_acc
	model_performance['BestTrain.Acc'] = best_train_acc
	model_performance['TimeCost'] = utils.time_cost(start_time)
	utils.config_save(args,model_performance)
        last_step = './save_model/'+args.save_path.split('/')[-1].split('.')[0]+'_laststep.ckpt'

        saver.save(sess,last_step)
	log.close()

if __name__ == '__main__' :
	train()




