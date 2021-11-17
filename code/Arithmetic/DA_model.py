import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from DNN_Attention import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
from tensorflow.keras.optimizers import Adamax, Nadam
import numpy as np
from tensorflow.keras import utils

def create_model(maxlen, vocab):
    '''
    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = weight_matrix / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(weight_matrix), axis=-1, keepdims=True)), K.floatx())
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(w_n.shape[0].eval())))
        return args.ortho_reg*reg
    '''
    # 词袋的大小
    # vocab_size = len(vocab)

    ##### Inputs #####
    # sentence_input = Input(shape=(maxlen,), dtype='int32', name='sentence_input')
    e_w = Input(shape=(maxlen, 3,  ), dtype='float32', name='sentence_input')
    print(e_w)
    # e_w = Change_dim(maxlen)(e_w)
    # print(e_w)
    # neg_input = Input(shape=(args.neg_size, maxlen), dtype='int32', name='neg_input')

    ##### Construct word embedding layer #####
    # 词嵌入层
    # word_emb = Embedding(6, 1, mask_zero=True, name='word_emb')

    ##### Compute sentence representation ####
    # e_w表示词的特征向量
    # e_w = word_emb(sentence_input)
    # print(e_w)
    # y_s表示这组向量的平均值
    

    y_s = Average(maxlen)(e_w)
    # print(y_s)
    # 使用注意力机制计算每个词在这个句子中的权重
    att_weights = Attention(name='att_weights')([e_w, y_s])
    # 获得句子向量，表示词向量的加权平均和
    z_s = WeightedSum()([e_w, att_weights])

    ##### Compute representations of negative instances #####
    # e_neg = word_emb(neg_input)
    # z_n = Average()(e_neg)

    ##### Reconstruction #####
    # p_t = Dense(args.aspect_size)(z_s)
    p_t = Dense(3)(z_s)
    p_t = Activation('relu', name='p_t')(p_t)
    # r_s = WeightedAspectEmb(args.aspect_size, args.emb_dim, name='aspect_emb',
            # W_regularizer=ortho_reg)(p_t)
    p_n = Dense(2)(p_t)
    p_n = Activation('softmax', name='p_n')(p_n)

    ##### Loss #####
    # loss = MaxMargin(name='max_margin')([z_s, z_n, r_s])
    model = Model(inputs=e_w, outputs=p_n)

    ### Word embedding and aspect embedding initialization ######
    '''
    if args.emb_path:
        from w2vEmbReader import W2VEmbReader as EmbReader
        emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
        logger.info('Initializing word embedding matrix')
        model.get_layer('word_emb').W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').W.get_value()))
        logger.info('Initializing aspect embedding matrix as centroid of kmean clusters')
        model.get_layer('aspect_emb').W.set_value(emb_reader.get_aspect_matrix(args.aspect_size))
    '''
    return model


if __name__ == '__main__':
    maxlen = 5
    vocab = {'I': 1, 'am': 2, 'a': 3, 'student': 4, 'how': 5, 'do': 6, 'you': 7}

    x_train = [[[1,2,3],[2,3,4],[3,4,5],[4,5,6],[3,5,2]],
    [[1,4,5],[5,3,3],[3,6,2],[1,4,2],[4,3,3]],
    [[4,2,3],[2,4,3],[1,5,2],[4,2,1],[3,6,1]]]


    # x_train = [[1,3,4,2,5], [4,6,3,1,2], [6,2,4,3,1]]
    # x_train = x_train.to_numpy()
    x_train = np.array(x_train)
    # print(x_train.shape)
    x_train = x_train[:,:,:,np.newaxis]
    # print(x_train)
    y_train = [1, 0, 1]
    y_train = utils.to_categorical(y_train, 2)
    # print(y_train)
    # print(type(x_train))
    # print(type(y_train))

    model = create_model(maxlen, vocab)

    OPTIMIZER = Nadam(learning_rate = 0.002, beta_1 = 0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs = 5)
    model.summary()





