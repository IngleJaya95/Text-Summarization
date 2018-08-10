
import os
os.system("export CUDA_VISIBLE_DEVICES=''")
import tensorflow as tf
import pickle # for loading and saving the file
import numpy as np # for mathematical op in python
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.layers import core as layers_core
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu




# If not interactive then uncomment it.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--encoder_hidden_units",type=int, help="The no. of hidden units your encoder should have",default=512)
parser.add_argument("--decoder_hidden_units",type=int, help="The no. of hidden units your decoder should have",default=512)
parser.add_argument("--batch",type=int, help="The no of element in a batch",default=20)
parser.add_argument("--maxEncoderTime",type=int, help="The maximum lenght encoder can have",default=150)
parser.add_argument("--maxDecoderTime",type=int, help="The maximum length decoder can have",default=150)
parser.add_argument("--max_gradient_norm",type=float, help="The clipping limit on the gradient",default=5.0)
parser.add_argument("--lr",type=float, help="Learning Rate",default=0.0001)
parser.add_argument("--epoch_1",type=int, help="Training Epochs using Training Helper",default=4)
parser.add_argument("--epoch_2",type=int, help="Training Epochs using Greedy Helper",default=20)
parser.add_argument("--val_step_shift",type=int, help="After how many step you wish to have a log entry in your log file",default=100)
parser.add_argument("--beamWidth",type = int, help="Set the beam width",default=0)
parser.add_argument("--beamSearch", type= bool, help="True : beamSearch enable, False : BeamSearch disable",default=False)
parser.add_argument("--con_name",type=str ,help="The name that make this experiment unique",default="Awesome_hehe")
parser.add_argument("--encoder_choice",type=int,help="0: Bi-directional Encoder, 1: Unidirectional Encoder",default=0)
parser.add_argument("--mode",type=int,help="0: basic Decoder, 1: basic decoder with attention, 2: hirarchial decoder",default=0)
parser.add_argument("--storingThreshold",type=float,help="The thresold bleu score after which we prefer to store",default=0.4)
parser.add_argument("--dropout",type=int,help="1: To apply dropout, 0: No dropout",default=0)
parser.add_argument("--dropval",type=float,help="Give the probability to drop",default=0.4)
args = parser.parse_args()




##############################################################
############# Hyper Parameter Section ########################
##############################################################
#1
encoder_hidden_units = args.encoder_hidden_units
#2
decoder_hidden_units = args.decoder_hidden_units

#3
batchSize = args.batch#20
#4
maxEncoderTime = args.maxEncoderTime #150
#5
maxDecoderTime = args.maxDecoderTime #150

# The maximum value a gradient can achieve. If more will be clipped
#6
max_gradient_norm = args.max_gradient_norm #5

# The Learning Rate
#6
learning_rate = args.lr #0.0001

#The Epochs : epoch_1 corrosponds to training using training Helper.
# epoch_2 corrosponds to training using greedy helper
#8
epoch_1 =  args.epoch_1 #2
#9
epoch_2 = args.epoch_2 #2

epochs = epoch_1 + epoch_2

#10
val_step_shift = args.val_step_shift #100 # After how many step you wish to have a log
sno = 0 # Just to make identification easier
#On which epoch wish to switch from trainHelper to greedy helper

# If bleu score is greater than the threshold then save the model
#11
storingThreshold = args.storingThreshold #0.4

# For the sake of beam Search
#11
beamWidth = args.beamWidth  #10
#12
beamSearch =args.beamSearch #False

# The name which respective indicative file will have in their respective directory
#13
con_name = args.con_name #"Trying"

# Bidirectional or Unidirectional
# 0: for the bidirectional
# 1: for the unidirectional
#14
encoder_choice = args.encoder_choice #0


attenType = "Dhoka"
# Validation size of the Batch
val_batch_size = batchSize

#mode =0 basic ,1 basic with attention ,hiearchical = 2 
#15
mode =  args.mode # 0

dropoutApplier = args.dropout
dropval = args.dropval
dpFixed = dropval
##############################################################################################


##############################################################
############# Hyper Parameter Section ########################
##############################################################
# encoder_hidden_units = 512
# decoder_hidden_units = 512

# batchSize = 50
# maxEncoderTime = 150
# maxDecoderTime = 150

# dropoutApplier = 1
# dropval = 0.8
# dpFixed = dropval
# # The maximum value a gradient can achieve. If more will be clipped
# max_gradient_norm = 5

# # The Learning Rate
# learning_rate = 0.0001

# #The Epochs : epoch_1 corrosponds to training using training Helper.
# # epoch_2 corrosponds to training using greedy helper
# epoch_1 = 1
# epoch_2 = 0

# epochs = epoch_1 + epoch_2
# val_step_shift = 100 # After how many step you wish to have a log
# sno = 0 # Just to make identification easier
# #On which epoch wish to switch from trainHelper to greedy helper

# # If bleu score is greater than the threshold then save the model
# storingThreshold = 0.4

# # For the sake of beam Search
# beamWidth = 10
# beamSearch = False

# # The name which respective indicative file will have in their respective directory
# con_name = "Trying"

# # Bidirectional or Unidirectional
# # 0: for the bidirectional
# # 1: for the unidirectional
# encoder_choice = 1


# attenType = "Dhoka"
# # Validation size of the Batch
# val_batch_size = batchSize

# #mode =0 basic ,1 basic with attention ,hiearchical = 2 
# mode = 0

#################################################################
################# Hyper Parameter Section ends here #############
#################################################################


# In[45]:


#################################################################
############ Support Code Section ###############################
#################################################################

# How to use : create a folder name "repo" in the directory
# where file is located
# obj : the object required to save
# name : the name for the host file of the data

def saveObject(obj,name): 
    pickle.dump(obj,open( "repo/data/core/"+name+".pkl", "wb" ))

def loadObject(name):
    obj = pickle.load( open( "repo/data/core/"+name+".pkl", "rb" ) )
    return obj

fileName = "repo/log/" +con_name+ str(sno) + "_Mode_" + str(mode) + "_batchSize_" + str(batchSize) +"_atype_" +attenType +"_epochs_" + str(epochs) + ".csv"
logFile = open(fileName,"w")
heading = "epoch ,steps ,train_loss ,val_loss ,bleu_score \n"
logFile.write(heading)
logFile.flush()


# In[54]:


##############################################################
########### Function Important ###############################
##############################################################

def enc_data_gen(file , pad_num):
    data = file.read()
    spl = data.split("\n")
    del spl[len(spl)-1]    # becoz last line is ''
    
    dat_mat = np.zeros([len(spl),150]) + pad_num        
    i = 0
    len_cont = [] # to contain the lengths of sequence
    for line in spl:
        textstr = line.split(' ')
        del textstr[len(textstr)-1]
        newLine = []
        for tex in textstr:
            if tex not in encT2N.keys():
                tex = '<unk>'
            textVal = encT2N[tex]
            newLine.append(textVal)
        length = len(newLine)
        dat_mat[i,:length] = np.array(newLine)
        len_cont.append(length) 
        i=i+1

# just for a better/powerful future 
    len_cont = np.array(len_cont)
    return dat_mat,len_cont

def dec_inp_gen(file , pad_num):
    data = file.read()
    spl = data.split("\n")
    del spl[len(spl)-1]    # becoz last line is ''
    
    dat_mat = np.zeros([len(spl),150]) +pad_num        
    i = 0
    len_cont = [] # to contain the lengths of sequence
    for line in spl:
        textstr = line.split(' ')
        del textstr[len(textstr)-1]
        newLine = []
        for tex in textstr:
            if tex not in outT2N.keys():
                tex = '<unk>'
            textVal = outT2N[tex]
            newLine.append(textVal)
        length = len(newLine)
        #It has to start from <GO>.
        dat_mat[i,0] = outT2N['<GO>']        
        dat_mat[i,1:length+1]= np.array(newLine)
        len_cont.append(length+1) 
        i=i+1

    # just for a better future 
    len_cont = np.array(len_cont)
    return dat_mat,len_cont

def dec_out_gen(file , pad_num):
    data = file.read()
    spl = data.split("\n")
    del spl[len(spl)-1]    # becoz last line is ''
    
    dat_mat = np.zeros([len(spl),150]) +pad_num        
    i = 0
    len_cont = [] # to contain the lengths of sequence
    for line in spl:
        textstr = line.split(' ')
        del textstr[len(textstr)-1]
        newLine = []
        for tex in textstr:
            if tex not in outT2N.keys():
                tex = '<unk>'
            textVal = outT2N[tex]
            newLine.append(textVal)
        length = len(newLine)
        dat_mat[i,:length]= np.array(newLine)
        #Since it has to end with 'End of String'
        dat_mat[i,length] = outT2N['<EOS>']
        len_cont.append(length+1)
        i=i+1

    # just for a better future 
    len_cont = np.array(len_cont)
    return dat_mat,len_cont

def target_w_gen(file):
    data = file.read()
    spl = data.split("\n")
    del spl[len(spl)-1]    # becoz last line is ''
    
    dat_mat = np.zeros([len(spl),150])        
    i = 0
    len_cont = [] # to contain the lengths of sequence
    for line in spl:
        textstr = line.split(' ')
        del textstr[len(textstr)-1]
        length = len(textstr)
        dat_mat[i,:length+1]= 1
        #Since it has to end with 'End of String'
        i=i+1
    # just for a better future 
    return dat_mat

def val_belu_ref_gen(file):
    data = file.read()
    spl = data.split("\n")
    del spl[len(spl)-1]    # becoz last line is ''
    texColl = [] # Contain All the string in tokken format
    for line in spl:
        textstr = line.split(' ')
        del textstr[len(textstr)-1]
        texColl.append(textstr)
    # just for a better future 
    return texColl



# The code that generate the data
def Datafeeder(step,batch_size,trainMet=True):
    #encSequenceLen feeder
    feedEncSeqLen = train_enc_inp_len[step*batch_size:(step+1)*batch_size]
    #decSequenceLen feeder
    feedDecSeqLen = train_dec_inp_len[step*batch_size:(step+1)*batch_size]

    enc_truncate_factor = max(feedEncSeqLen)
    dec_truncate_factor = max(feedDecSeqLen)

    #Encoder Input feeder
    enc_fd = train_enc_inp_mat[step*batch_size:(step+1)*batch_size]
    enc_tfd = enc_fd[:,:enc_truncate_factor]
    feedEncInput = (enc_tfd).T # Transpose since Time major format

    #Decoder Input feeder
    dec_fd = train_dec_inp_mat[step*batch_size:(step+1)*batch_size]
    dec_tfd = dec_fd[:,:dec_truncate_factor]
    feedDecInput = (dec_tfd).T

    #Decoder Output Feeder
    dec_fd = train_dec_out_mat[step*batch_size:(step+1)*batch_size]
    dec_tfd = dec_fd[:,:dec_truncate_factor]
    feedDecOutput = (dec_tfd).T

    #TargetWeights Feeder
    dec_fd = train_target_w_mat[step*batch_size:(step+1)*batch_size]
    dec_tfd = dec_fd[:,:dec_truncate_factor]
    feedTargetW = (dec_tfd).T

    inputData = {encoder_inputs:feedEncInput,decoder_inputs:feedDecInput,decoder_outputs:feedDecOutput,
             target_weights:feedTargetW,enc_seqLen:feedEncSeqLen,dec_seqLen:feedDecSeqLen,choice:trainMet,
                max_dec_len : dec_truncate_factor}
    
    return inputData



def val_Datafeeder(step,batch_size,trainMet=True):
    #encSequenceLen feeder
    feedEncSeqLen = val_enc_inp_len[step*batch_size:(step+1)*batch_size]
    #decSequenceLen feeder
    feedDecSeqLen = val_dec_inp_len[step*batch_size:(step+1)*batch_size]

    enc_truncate_factor = max(feedEncSeqLen)
    dec_truncate_factor = max(feedDecSeqLen)

    #Encoder Input feeder
    enc_fd = val_enc_inp_mat[step*batch_size:(step+1)*batch_size]
    enc_tfd = enc_fd[:,:enc_truncate_factor]
    feedEncInput = (enc_tfd).T # Transpose since Time major format

    #Decoder Input feeder
    dec_fd = val_dec_inp_mat[step*batch_size:(step+1)*batch_size]
    dec_tfd = dec_fd[:,:dec_truncate_factor]
    feedDecInput = (dec_tfd).T

    #Decoder Output Feeder
    dec_fd = val_dec_out_mat[step*batch_size:(step+1)*batch_size]
    dec_tfd = dec_fd[:,:dec_truncate_factor]
    feedDecOutput = (dec_tfd).T

    #TargetWeights Feeder
    dec_fd = val_target_w_mat[step*batch_size:(step+1)*batch_size]
    dec_tfd = dec_fd[:,:dec_truncate_factor]
    feedTargetW = (dec_tfd).T

    inputData = {encoder_inputs:feedEncInput,decoder_inputs:feedDecInput,decoder_outputs:feedDecOutput,
             target_weights:feedTargetW,enc_seqLen:feedEncSeqLen,dec_seqLen:feedDecSeqLen,choice:trainMet,
                max_dec_len : dec_truncate_factor}
    
    return inputData




def test_Datafeeder(step,batch_size,trainMet=True):
    #encSequenceLen feeder
    feedEncSeqLen = test_enc_inp_len_enp[step*batch_size:(step+1)*batch_size]
    #decSequenceLen feeder
    feedDecSeqLen = test_enc_inp_len_enp[step*batch_size:(step+1)*batch_size]

    enc_truncate_factor = int(max(feedEncSeqLen))
    dec_truncate_factor = int(max(feedDecSeqLen))

    #Encoder Input feeder
    enc_fd = test_enc_inp_mat_enp[step*batch_size:(step+1)*batch_size]
    enc_tfd = enc_fd[:,:enc_truncate_factor]
    feedEncInput = (enc_tfd).T # Transpose since Time major format

    #Decoder Input feeder
    dec_fd = test_enc_inp_mat_enp[step*batch_size:(step+1)*batch_size]
    dec_tfd = dec_fd[:,:dec_truncate_factor]
    feedDecInput = (dec_tfd).T

    #Decoder Output Feeder
    dec_fd = test_enc_inp_mat_enp[step*batch_size:(step+1)*batch_size]
    dec_tfd = dec_fd[:,:dec_truncate_factor]
    feedDecOutput = (dec_tfd).T

    #TargetWeights Feeder
    dec_fd = test_enc_inp_mat_enp[step*batch_size:(step+1)*batch_size]
    dec_tfd = dec_fd[:,:dec_truncate_factor]
    feedTargetW = (dec_tfd).T

    inputData = {encoder_inputs:feedEncInput,decoder_inputs:feedDecInput,decoder_outputs:feedDecOutput,
             target_weights:feedTargetW,enc_seqLen:feedEncSeqLen,dec_seqLen:feedDecSeqLen,choice:trainMet,
                max_dec_len : dec_truncate_factor}
    
    return inputData






# For Generating Text for a batch
def GenerateText(genText):
    genText = np.array(genText)
    # MaxTime * BatchSize is for systems, I love the transpose.
    # Standard way to represent the text
    genText = genText.T
    lines,maxWords = np.shape(genText)

    collectionText = []

    for i in range(lines):
        line_with_no = genText[i]
        line_text = []
        for word_no in line_with_no:
            word_text = outN2T[word_no]
            if word_text != "<PAD>" and word_text != "<unk>":
                if word_text == "<EOS>":
                    break
                line_text.append(word_text)
        l_text = " ".join(line_text)
        collectionText.append(l_text)
    return collectionText


# In[47]:





#Creating PADDING, EOS and GO

key1 = '<PAD>'
key2 = '<EOS>'
key3 = '<GO>'

val1 = np.array([np.zeros(256)])
val2 = np.array([np.zeros(256)-0.5])
val3 = np.array([np.zeros(256)+0.5])


# In[48]:


inEmb = loadObject("inpEmb")
# inEmb is a dictionary 
# that contain the input embedding : embedding
# A text to no disctionary : encT2N
# A no to text disctionary : encN2T

encN2T = inEmb['encT2N']
encT2N = inEmb['encN2T']
embInp = inEmb['embedding']

Len = len(encN2T)
encN2T[Len] = key1
encN2T[Len+1] = key2
encN2T[Len+2] = key3
encT2N[key1] = Len
encT2N[key2] = Len + 1
encT2N[key3] = Len + 2
embInp = np.append(embInp,val1,0)
embInp = np.append(embInp,val2,0)
embInp = np.append(embInp,val3,0)
enc_vocab_size = len(encT2N)

#Embedding Exchanged
currentIndex = encT2N["<PAD>"] 
targetIndex = 0

currentKey = "<PAD>"
targetKey = encN2T[targetIndex]

#Exchange The embeddings
tempEmb = np.zeros([1,256])
tempEmb[0,:] = embInp[currentIndex,:]
embInp[currentIndex,:] = embInp[targetIndex,:]
embInp[targetIndex,:] = tempEmb[0,:]

#Exchange keys and Value
encT2N[targetKey] = currentIndex
encT2N[currentKey] = targetIndex

encN2T[currentIndex] = targetKey
encN2T[targetIndex] = currentKey


# In[49]:


outEmb = loadObject("outEmb")
# outEmb is a dictionary 
# that contain the input embedding : embedding
# A text to no disctionary : encT2N
# A no to text disctionary : encN2T

outN2T = outEmb['outT2N']
outT2N = outEmb['outN2T']
embOut = outEmb['embedding']

Len = len(outN2T)
outN2T[Len] = key1
outN2T[Len+1] = key2
outN2T[Len+2] = key3
outT2N[key1] = Len
outT2N[key2] = Len + 1
outT2N[key3] = Len + 2
embOut = np.append(embOut,val1,0)
embOut = np.append(embOut,val2,0)
embOut = np.append(embOut,val3,0)

#Fixing Padding Issue
currentIndex = outT2N["<PAD>"]
targetIndex = 0

currentKey = '<PAD>'
targetKey = outN2T[0]

#Fixing the embedding matrix
temp = np.zeros([1,256])
temp[0,:] = embOut[currentIndex,:]
embOut[currentIndex,:] = embOut[targetIndex,:]
embOut[targetIndex,:] = temp[0,:]

#Fixing the dictionary
outT2N[currentKey] = targetIndex
outT2N[targetKey] = currentIndex

outN2T[targetIndex] = currentKey
outT2N[currentIndex] = targetKey

dec_vocab_size = len(outN2T)


# In[50]:


#Reading the Data

DataLoc = 'repo/data/'

#TrainData
train_path = DataLoc + 'train/'
train_summariesData = open(train_path+'summaries.txt','r')
train_tableData = open(train_path+'train.combined','r')

#Formatting Training Data
train_enc_inp_mat,train_enc_inp_len=enc_data_gen(train_tableData, encT2N['<PAD>'])
train_dec_inp_mat,train_dec_inp_len=dec_inp_gen(train_summariesData, outT2N['<PAD>'])

train_summariesData = open(train_path+'summaries.txt','r')
train_dec_out_mat,train_dec_out_len=dec_out_gen(train_summariesData, outT2N['<PAD>'])

train_summariesData = open(train_path+'summaries.txt','r')
train_target_w_mat = target_w_gen(train_summariesData)

#ValidationData
val_path = DataLoc + 'dev/'

# Formatting Validation Data in the required Form
val_tableData = open(val_path+'dev.combined','r')
val_enc_inp_mat,val_enc_inp_len=enc_data_gen(val_tableData, encT2N['<PAD>'])

val_summariesData = open(val_path+'summaries.txt','r')
val_dec_inp_mat,val_dec_inp_len=dec_inp_gen(val_summariesData, outT2N['<PAD>'])

val_summariesData = open(val_path+'summaries.txt','r')
val_dec_out_mat,val_dec_out_len=dec_out_gen(val_summariesData, outT2N['<PAD>'])

val_summariesData = open(val_path+'summaries.txt','r')
val_target_w_mat = target_w_gen(val_summariesData)

val_summariesData = open(val_path+'summaries.txt','r') # For creating reference for BLEU Score
val_belu_ref_text = val_belu_ref_gen(val_summariesData)

# #Formating Validation Data
# valInp_data_mat,valInp_len=data_gen(val_tableData, encT2N['<PAD>'])
# valTarget_data_mat,valTarget_len=data_gen(val_summariesData, outT2N['<PAD>'])

#TestingData
test_path = DataLoc + 'test/'
test_tableData = open(test_path+'test.combined','r')

#Formatting Test Data
test_enc_inp_mat,test_enc_inp_len=enc_data_gen(test_tableData, encT2N['<PAD>'])

test_enc_inp_mat_enp = np.zeros([4000,150])
len_testing,w_testing = np.shape(test_enc_inp_mat)
test_enc_inp_mat_enp[:len_testing,:] = test_enc_inp_mat[:,:]

test_enc_inp_len_enp = np.zeros(4000) + 150
test_enc_inp_len_enp[:len_testing] = test_enc_inp_len


train_dp_count = len(train_enc_inp_len)
val_dp_count = len(val_dec_inp_len)
test_act_count = len(test_enc_inp_len)
test_dp_count = len(test_enc_inp_len)


# # Graph

# In[51]:


########################################################################
################### Graph #############################################
#######################################################################
dropoutApplier = 1
dropval = 0.8
#Reseting the graph
tf.reset_default_graph()
tf.set_random_seed(1234)

#Loading Embedding Matrix
embeddingMatrixOut = tf.Variable(embOut, dtype=tf.float32,name="embeddingMatrixOut")
embeddingMatrixInp = tf.Variable(embInp, dtype=tf.float32,name="embeddingMatrixInp")

#Placeholders
encoder_inputs = tf.placeholder(shape=(None,batchSize),dtype=tf.int32,name="encoder_inputs")
decoder_inputs = tf.placeholder(shape=(None,batchSize),dtype=tf.int32,name="decoder_inputs")
decoder_outputs = tf.placeholder(shape=(None,batchSize),dtype=tf.int32,name="decoder_outputs")


target_weights = tf.placeholder(shape=(None,batchSize),dtype=tf.float32,name="target_weights")

enc_seqLen = tf.placeholder(shape=(batchSize),dtype=tf.int32,name="enc_seqLen")
dec_seqLen = tf.placeholder(shape=(batchSize),dtype=tf.int32,name="dec_seqLen")

#Embedding Lookups
enc_embedded_input = tf.nn.embedding_lookup(embeddingMatrixInp, encoder_inputs)
dec_embedded_input = tf.nn.embedding_lookup(embeddingMatrixOut, decoder_inputs)

##############################################################
############### Encoder ######################################
#############################################################3#

#Creating Cell for the Encoder

#Encoder Choice 0: for the bidirectional, 1: for the unidirectional

if encoder_choice == 0:
    #Calling Dynamic RNN
    enc_forward_cell = LSTMCell(encoder_hidden_units)
    enc_backward_cell = LSTMCell(encoder_hidden_units)
    if dropoutApplier == 1:
        enc_forward_cell = tf.contrib.rnn.DropoutWrapper(
            enc_forward_cell, output_keep_prob=1-dropval)
        enc_backward_cell = tf.contrib.rnn.DropoutWrapper(
            enc_backward_cell, output_keep_prob=1-dropval)
    ((encoder_fw_outputs,
      encoder_bw_outputs),
     (encoder_fw_final_state,
      encoder_bw_final_state)) = (
        tf.nn.bidirectional_dynamic_rnn(cell_fw = enc_forward_cell,
                                        cell_bw = enc_backward_cell,
                                        inputs = enc_embedded_input,
                                        sequence_length = enc_seqLen,
                                        dtype = tf.float32,
                                        time_major = True)
        )

    #Concatatenating Forward and Backward output
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)


    encoder_final_state_c = tf.concat(
        (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

    encoder_final_state_h = tf.concat(
        (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

    encoder_final_state = LSTMStateTuple(
        c=encoder_final_state_c,
        h=encoder_final_state_h
    )

else:
    encoder_cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_units)
    if dropoutApplier == 1:
        encoder_cell = tf.contrib.rnn.DropoutWrapper(
            encoder_cell, output_keep_prob=1-dropval)
    initial_state = encoder_cell.zero_state(batchSize,dtype =tf.float32)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, enc_embedded_input, initial_state = initial_state,dtype = tf.float32, 
    sequence_length=enc_seqLen, time_major=True)
    
# convW = tf.Variable(tf.random_uniform([2,2*encoder_hidden_units, decoder_hidden_units], -1, 1), dtype=tf.float32,name="convW")
# convb = tf.Variable(tf.zeros([decoder_hidden_units]), dtype=tf.float32,name="convb")
# #convAct = tf.add(tf.matmul(encoder_hidden_units, convW), convb)
# #initDecoder = tf.tanh(convAct)
# tf.matmul(convW,encoder_final_state)

# Decoder
if mode == 1:
    attention_states = tf.transpose(encoder_outputs,[1,0,2])
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(decoder_hidden_units,attention_states,memory_sequence_length=enc_seqLen)
    
    decoder_cell = LSTMCell(decoder_hidden_units)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,attention_mechanism,attention_layer_size = 2*decoder_hidden_units)
    encoder_final_state = decoder_cell.zero_state(dtype =tf.float32,batch_size = batchSize) 
else:
    if encoder_choice == 0:
        decoder_cell = LSTMCell(2*decoder_hidden_units)
    else:
        decoder_cell = LSTMCell(decoder_hidden_units)


projection_layer = layers_core.Dense(dec_vocab_size, use_bias=False, name="output_projection")

# Attention Is All You Need



#Training Helper
helper_1 = tf.contrib.seq2seq.TrainingHelper(inputs = dec_embedded_input,sequence_length = dec_seqLen,time_major=True)
#Normal Decoder


decoder_1 = tf.contrib.seq2seq.BasicDecoder(decoder_cell,helper_1,encoder_final_state,output_layer = projection_layer)
#Dynamic Decoding
outputs_1, final_context_state_1, _= tf.contrib.seq2seq.dynamic_decode(decoder_1,swap_memory=True,output_time_major=True)
logits_1 = outputs_1.rnn_output
#Calculating Cross Entropy
crossent_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs,logits = logits_1)
#Calculation the loss_1 the so called Train Helper Loss
loss_1 = (tf.reduce_sum(crossent_1*target_weights)/batchSize)




#Greedy Helper
max_dec_len = tf.placeholder(tf.int32,shape=(), name="max_dec_len")

tgt_sos_id = outT2N['<GO>']
tgt_eos_id = outT2N['<EOS>']

helper_2 = tf.contrib.seq2seq.GreedyEmbeddingHelper(
    embeddingMatrixOut,
    tf.fill([batchSize],tgt_sos_id),tgt_eos_id)

#Inferential Decoder
inference_decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell,
    helper_2,
    encoder_final_state,
    output_layer=projection_layer)

#Dynamic Decoder
outputs_2, final_context_state_infer_2, _= tf.contrib.seq2seq.dynamic_decode(
    inference_decoder,
    maximum_iterations=max_dec_len,
    output_time_major=True)

translations = outputs_2.sample_id
#Calculating Logits for greedy helper

####


logits_2 = outputs_2.rnn_output

#Adding Padding to the logits
logit_shape = tf.shape(logits_2)
rem = max_dec_len - logit_shape[0]
paddings = [[0, rem],[0,0],[0,0]]
logits_infer = tf.pad(logits_2,paddings,'CONSTANT')

# Calculating the cross Entropy
crossent_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs,logits = logits_infer)
#Calculating the loss for greedy Helper type
loss_2 = (tf.reduce_sum(crossent_2*target_weights)/batchSize)

choice = tf.placeholder(dtype=tf.bool)

#NotHere 
################################################
#Beamer Search

# decoder_initial_state = tf.contrib.seq2seq.tile_batch(
# encoder_final_state,
# multiplier = 10,
# )

# decoder_3 = tf.contrib.seq2seq.BeamSearchDecoder(
# cell = decoder_cell,
# embedding = embeddingMatrixOut,
# start_tokens = tgt_sos_id,
# end_tokens = tgt_eos_id,
# intial_state = decoder_initial_state,
# beam_width = 10,
# output_layer = projection_layer,
# length_penalty_weight = 0.0,
# )

# outputs_3, final_context_state_infer_3, _= tf.contrib.seq2seq.dynamic_decode(
#     decoder_3,
#     maximum_iterations=max_dec_len,
#     output_time_major=True)

# translations = outputs_3.sample_id

# logits_3 = outputs_3.rnn_output

# #Adding Padding to the logits
# logit_shape_3 = tf.shape(logits_3)
# rem_3 = max_dec_len - logit_shape_3[0]
# paddings_3 = [[0, rem_3],[0,0],[0,0]]
# logits_infer_3 = tf.pad(logits_3,paddings_3,'CONSTANT')

# # Calculating the cross Entropy
# crossent_3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs_3,logits = logits_infer_3)
# #Calculating the loss for greedy Helper type
# loss_3 = (tf.reduce_sum(crossent_3*target_weights)/batchSize)



####



train_loss = loss_1#tf.cond(choice,lambda: loss_1,lambda: loss_2)
val_loss = loss_2

#Calculate and Clip Gradient 
params = tf.trainable_variables()
gradients = tf.gradients(train_loss,params)
clipped_gradients,_ = tf.clip_by_global_norm(gradients,max_gradient_norm)

#Optimization

optimizer = tf.train.AdamOptimizer(learning_rate)
update_step = optimizer.apply_gradients(zip(clipped_gradients,params))

################################ For loss_2 ############
gradient_2 = tf.gradients(val_loss,params)
clipped_gradient_2,_ = tf.clip_by_global_norm(gradient_2,max_gradient_norm)

#Optimization

optimizer_2 = tf.train.AdamOptimizer(learning_rate)
update_step_2 = optimizer_2.apply_gradients(zip(clipped_gradient_2,params))




saver = tf.train.Saver()
encoder_inputs


# In[52]:


#initializing the Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[56]:


modelSavedOnce = 0
#No of times whole data need to be seen by network
bestScore = -1

#In how many partition whole data will be seen by the network
no_of_train_steps = train_dp_count//batchSize
no_of_val_steps = val_dp_count//val_batch_size

#Training Loop
for i in range(epoch_1):
    total_step_loss = 0
    for step in range(no_of_train_steps):
        trainMet = True
#         if i >= switching_epoch:
#             trainMet = False
        feedData = Datafeeder(step,batchSize,trainMet)
        _,stepLoss = sess.run([update_step,train_loss],feed_dict=feedData)
        total_step_loss += stepLoss
        print(step)
        
        if step % val_step_shift == 0:
            dropval = 0
            val_loss_total = 0.0
            #Going for the joy ride of Validation
            actualTextList = []
            #First need Data feeder. Haha! I have that.
            for valStep in range(no_of_val_steps):
                print(valStep," Started \n")
                # Get the validation data
                feedValData = val_Datafeeder(valStep,batchSize)
                # Loss 2 is for the validation, which can be further add into second part training
                valStepLoss,genRatedText = sess.run([val_loss,translations],feed_dict=feedValData)
                actualTextList += GenerateText(genRatedText)
                val_loss_total += valStepLoss
            #Every new score will be started from the zero.
            score = 0
            for s1,s2 in zip(val_belu_ref_text,actualTextList):
                score += sentence_bleu(s1, s2,weights=(0.25, 0.25, 0.25, 0.25))
            score = score/(len(val_belu_ref_text))            
            if score > storingThreshold:
                if score > bestScore:
                    bestScore = score
                    save_path = saver.save(sess, "repo/model/savedModel.ckpt")
                    modelSavedOnce = 1
            norm_train_loss = total_step_loss/val_step_shift
            norm_val_loss = val_loss_total/no_of_val_steps
            print("epoch ",i," Training Loss :",norm_train_loss," Validation Loss :", norm_val_loss,", belu : ",score)
            logFile.write(str(i) + "," + str(step) + "," + str(norm_train_loss) + ","+ str(norm_val_loss) + ","+str(score)+ "\n")
            logFile.flush()
            total_step_loss = 0
            val_loss_total = 0

            dropval = dpFixed 

for i in range(epoch_1,epoch_1+epoch_2):
    total_step_loss = 0
    for step in range(no_of_train_steps):
        trainMet = True
#         if i >= switching_epoch:
#             trainMet = False
        feedData = Datafeeder(step,batchSize,trainMet)
        _,stepLoss = sess.run([update_step_2,train_loss],feed_dict=feedData)
        total_step_loss += stepLoss
        print(step)
        
        if step % val_step_shift == 0:
            dropval = 0
            val_loss_total = 0.0
            #Going for the joy ride of Validation
            actualTextList = []
            #First need Data feeder. Haha! I have that.
            for valStep in range(no_of_val_steps):
                print(valStep," Started \n")
                # Get the validation data
                feedValData = val_Datafeeder(valStep,batchSize)
                # Loss 2 is for the validation, which can be further add into second part training
                valStepLoss,genRatedText = sess.run([val_loss,translations],feed_dict=feedValData)
                actualTextList += GenerateText(genRatedText)
                val_loss_total += valStepLoss
            #score = corpus_bleu(val_belu_ref_text, actualTextList, weights=(0.25,0.25,0.25,0.25))
            score = 0
            for s1,s2 in zip(val_belu_ref_text,actualTextList):
                score += sentence_bleu(s1, s2,weights=(0.25, 0.25, 0.25, 0.25))
            score = score/(len(val_belu_ref_text))            
            if score > storingThreshold:
                if score > bestScore:
                    bestScore = score
                    save_path = saver.save(sess, "repo/model/savedModel.ckpt")
                    modelSavedOnce = 1
            norm_train_loss = total_step_loss/val_step_shift
            norm_val_loss = val_loss_total/no_of_val_steps
            print("epoch ",i," Training Loss :",norm_train_loss," Validation Loss :", norm_val_loss,", belu : ",score)
            logFile.write(str(i) + "," + str(step) + "," + str(norm_train_loss) + ","+ str(norm_val_loss) + ","+str(score)+ "\n")
            logFile.flush()
            #print(total_step_loss/100)
            total_step_loss = 0
            val_loss_total = 0
            dropval = dpFixed

if modelSavedOnce == 0:
    save_path = saver.save(sess, "repo/model/savedModel.ckpt")
            
logFile.close()

no_of_test_steps = int(np.ceil(test_dp_count/batchSize))

actualTextList = []



for testStep in range(no_of_test_steps):
    print(testStep," Started \n")
    feedValData = test_Datafeeder(testStep,batchSize)
    genRatedText = sess.run(translations,feed_dict=feedValData)
    actualTextList += GenerateText(genRatedText)

fileText = open("repo/"+ con_name +"gen.txt","w")
for li in range(test_dp_count):
    fileText.write(actualTextList[li]+"\n")
fileText.close()

