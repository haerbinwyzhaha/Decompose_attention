# -*-coding:utf-8
import tensorflow as tf

class Decompose_Model():

	def __init__(self,parameters):
		self._init_parameters(parameters)
		self._build_graph()

	def _build_graph(self):
		self._init_placeholder()
		self.embedding = tf.Variable(self.embedding_,trainable=False)
		projected_input1,projected_input2 = self._embedding_block()
		beta,alpha = self._attend_block('attend',projected_input1,projected_input2)
		v1,v2 = self._compare_block('compare',beta,alpha,projected_input1,projected_input2)
		self.logits = self._aggregate_block('aggregate',v1,v2)
		self.loss = self._loss_op()
		self.train = self._train_op()
		self.pred = self._pred_op()
		self.acc = self._acc_op()

	def _init_parameters(self,parameters):
		self.hidden_size = parameters['hidden_size']
		self.embedding_size = parameters['embedding_size']
		# self.lr = parameters['learning_rate']
		self.classes = parameters['classes']
		# self.dropout = parameters['dropout']
		self.clip_value = parameters['clip_value']
		self.maxlen = parameters['maxlen']
		self.vocabulary_size = parameters['vocabulary_size']

	def _init_placeholder(self):
		self.input1 = tf.placeholder(dtype=tf.int32,shape=[None,self.maxlen])
		self.input2 = tf.placeholder(dtype=tf.int32,shape=[None,self.maxlen])
		self.length1 = tf.placeholder(dtype=tf.int32,shape=[None])
		self.length2 = tf.placeholder(dtype=tf.int32,shape=[None])
		self.learning_rate = tf.placeholder(dtype=tf.float32)
		self.dropout_rate = tf.placeholder(dtype=tf.float32)
		self.label = tf.placeholder(dtype=tf.int32,shape=[None])
		self.embedding_ = tf.placeholder(dtype=tf.float32,shape=[self.vocabulary_size,self.embedding_size])

	def _project_layer(self,inputs,hidden_size,reuse_flag=False):
		with tf.variable_scope('project',reuse=reuse_flag):
			initializer = tf.random_normal_initializer(0.0,0.1)
			outputs = tf.layers.dense(inputs,hidden_size,kernel_initializer=initializer)
		return outputs

	def _forward_layer(self,name,inputs,hidden_size,reuse_flag=False):
		with tf.variable_scope(name,reuse=reuse_flag):
			initializer = tf.random_normal_initializer(0.0,0.1)	
			with tf.variable_scope('layer1'):
				inputs = tf.nn.dropout(inputs,self.dropout_rate)
				result1 = tf.layers.dense(inputs,hidden_size,tf.nn.relu,kernel_initializer=initializer)
			
			with tf.variable_scope('layer2'):
				result1 = tf.nn.dropout(result1,self.dropout_rate)
				result2 = tf.layers.dense(result1,hidden_size,tf.nn.relu,kernel_initializer=initializer)
		return result2

	def _masked_softmax(self,inputs,length):
		# length : [batchsize]
		sequence_mask = tf.sequence_mask(length,maxlen=tf.shape(inputs)[-1],dtype=tf.float32)
		# sequence_mask : [batchsize,maxlen]
		sequence_mask_pad = tf.expand_dims(sequence_mask,1)
		# sequence_mask_pad : [batchsize,1,maxlen]
		max_item = tf.reduce_max(inputs,axis=-1,keepdims=True)
		mask_exp_input = tf.exp(inputs-max_item) * sequence_mask_pad
		mask_exp_sum = tf.reduce_sum(mask_exp_input,axis=-1,keepdims=True)
		
		result = mask_exp_input/mask_exp_sum
		return result

	def _embedding_block(self):
		with tf.name_scope('embedding_layer'):
			embedded_input1 = tf.nn.embedding_lookup(self.embedding,self.input1)
			embedded_input2 = tf.nn.embedding_lookup(self.embedding,self.input2)
		with tf.name_scope('project_layer'):
			projected_input1 = self._project_layer(embedded_input1,self.hidden_size)
			projected_input2 = self._project_layer(embedded_input2,self.hidden_size,reuse_flag=True)
		return projected_input1,projected_input2

	def _attend_block(self,name,input1,input2):
		with tf.variable_scope(name):
			f_input1 = self._forward_layer('F',input1,self.hidden_size)
			# f_input1 : [batchsize,maxlen,hidden_size]
			f_input2 = self._forward_layer('F',input2,self.hidden_size,reuse_flag=True)
			# f_input2 : [batchsize,maxlen,hidden_size]
			attended = tf.matmul(f_input1,tf.transpose(f_input2,[0,2,1]))
			# attended : [batchsize,input1_len,input2_len]
			attended_input1 = self._masked_softmax(tf.transpose(attended,[0,2,1]),self.length1)
			# attended_input1 : [batchsize,input2_len,input1_len]
			attended_input2 = self._masked_softmax(attended,self.length2)
			# attended_input2 : [batchsize,input1_len,input2_len]
			beta = tf.matmul(attended_input1,f_input1)
			# beta : [batchsize,input2_len,hidden_size]
			alpha = tf.matmul(attended_input2,f_input2)
			# alpha : [batchsize,input1_len,hidden_size]
		return beta,alpha

	def _compare_block(self,name,beta,alpha,projected_input1,projected_input2):
		with tf.variable_scope(name):
			input1 = tf.concat([alpha,projected_input1],axis=-1)
			input2 = tf.concat([beta,projected_input2],axis=-1)
			v1 = self._forward_layer('G',input1,self.hidden_size)
			v2 = self._forward_layer('G',input2,self.hidden_size,reuse_flag=True)

		return v1,v2

	def _aggregate_block(self,name,v1,v2):
		with tf.variable_scope(name):
			mask_v1 = tf.expand_dims(tf.sequence_mask(self.length1,maxlen=tf.shape(v1)[1],dtype=tf.float32),dim=2)
			mask_v2 = tf.expand_dims(tf.sequence_mask(self.length2,maxlen=tf.shape(v1)[1],dtype=tf.float32),dim=2)

			masked_v1 = v1*mask_v1
			masked_v2 = v2*mask_v2

			sum_v1 = tf.reduce_sum(masked_v1,axis=1)
			sum_v2 = tf.reduce_sum(masked_v2,axis=1)
			sum_all = tf.concat([sum_v1,sum_v2],axis=1)
			outputs = self._forward_layer('H',sum_all,self.hidden_size)
			logits = tf.layers.dense(outputs,self.classes,name='logits')
		return logits

	def _loss_op(self):
		with tf.name_scope('loss_block'):
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label,logits=self.logits)
			loss = tf.reduce_mean(losses,name='loss')
		return loss

	def _train_op(self):
		with tf.name_scope('train_block'):		
			optimizer = tf.train.AdagradOptimizer(self.learning_rate)
			gradients , v = zip(*optimizer.compute_gradients(self.loss))
			if self.clip_value != None :
				gradients, _ = tf.clip_by_global_norm(gradients,self.clip_value)
			train_op = optimizer.apply_gradients(zip(gradients,v))
		return train_op

	def _pred_op(self):
		with tf.name_scope('predict_block'):
			pred = tf.argmax(tf.nn.softmax(self.logits),axis = 1)
		return tf.cast(pred,dtype=tf.int32)


	def _acc_op(self):
		with tf.name_scope('accuracy_block'):
			equal_items = tf.cast(tf.equal(self.pred,self.label),dtype=tf.float32)
			accuracy = tf.reduce_mean(equal_items)
		return accuracy

	def feed_dict(self,input1,input2,length1,length2,label,lr,dropout):
		feed_dict = {
		self.input1:input1,
		self.input2:input2,
		self.length1:length1,
		self.length2:length2,
		self.label:label,
		self.learning_rate : lr,
		self.dropout_rate : dropout 
		}
		return feed_dict

