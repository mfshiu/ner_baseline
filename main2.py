import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import flat_classification_report

file_path='data/train_2.txt'


def loadInputFile(file_path):
    trainingset = list()  # store trainingset [content,content,...]
    position = list()  # store position [article_id, start_pos, end_pos, entity_text, entity_type, ...]
    mentions = dict()  # store mentions[mention] = Type
    with open(file_path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    datas = file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data = data.split('\n')
        content = data[0]
        trainingset.append(content)
        annotations = data[1:]
        for annot in annotations[1:]:
            annot = annot.split('\t')  # annot= article_id, start_pos, end_pos, entity_text, entity_type
            position.extend(annot)
            mentions[annot[3]] = annot[4]

    return trainingset, position, mentions


def CRFFormatData(trainingset, position, path):
    if (os.path.isfile(path)):
        os.remove(path)
    outputfile = open(path, 'a', encoding='utf-8')

    # output file lines
    count = 0  # annotation counts in each content
    tagged = list()
    for article_id in range(len(trainingset)):
        trainingset_split = list(trainingset[article_id])
        while '' or ' ' in trainingset_split:
            if '' in trainingset_split:
                trainingset_split.remove('')
            else:
                trainingset_split.remove(' ')
        start_tmp = 0
        for position_idx in range(0, len(position), 5):
            if int(position[position_idx]) == article_id:
                count += 1
                if count == 1:
                    start_pos = int(position[position_idx + 1])
                    end_pos = int(position[position_idx + 2])
                    entity_type = position[position_idx + 4]
                    if start_pos == 0:
                        token = list(trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token_idx == 0:
                                label = 'B-' + entity_type
                            else:
                                label = 'I-' + entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)

                    else:
                        token = list(trainingset[article_id][0:start_pos])
                        whole_token = trainingset[article_id][0:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue

                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            outputfile.write(output_str)

                        token = list(trainingset[article_id][start_pos:end_pos])
                        whole_token = trainingset[article_id][start_pos:end_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            # BIO states
                            if token[0] == '':
                                if token_idx == 1:
                                    label = 'B-' + entity_type
                                else:
                                    label = 'I-' + entity_type
                            else:
                                if token_idx == 0:
                                    label = 'B-' + entity_type
                                else:
                                    label = 'I-' + entity_type

                            output_str = token[token_idx] + ' ' + label + '\n'
                            outputfile.write(output_str)

                    start_tmp = end_pos
                else:
                    start_pos = int(position[position_idx + 1])
                    end_pos = int(position[position_idx + 2])
                    entity_type = position[position_idx + 4]
                    if start_pos < start_tmp:
                        continue
                    else:
                        token = list(trainingset[article_id][start_tmp:start_pos])
                        whole_token = trainingset[article_id][start_tmp:start_pos]
                        for token_idx in range(len(token)):
                            if len(token[token_idx].replace(' ', '')) == 0:
                                continue
                            output_str = token[token_idx] + ' ' + 'O' + '\n'
                            outputfile.write(output_str)

                    token = list(trainingset[article_id][start_pos:end_pos])
                    whole_token = trainingset[article_id][start_pos:end_pos]
                    for token_idx in range(len(token)):
                        if len(token[token_idx].replace(' ', '')) == 0:
                            continue
                        # BIO states
                        if token[0] == '':
                            if token_idx == 1:
                                label = 'B-' + entity_type
                            else:
                                label = 'I-' + entity_type
                        else:
                            if token_idx == 0:
                                label = 'B-' + entity_type
                            else:
                                label = 'I-' + entity_type

                        output_str = token[token_idx] + ' ' + label + '\n'
                        outputfile.write(output_str)
                    start_tmp = end_pos

        token = list(trainingset[article_id][start_tmp:])
        whole_token = trainingset[article_id][start_tmp:]
        for token_idx in range(len(token)):
            if len(token[token_idx].replace(' ', '')) == 0:
                continue

            output_str = token[token_idx] + ' ' + 'O' + '\n'
            outputfile.write(output_str)

        count = 0

        output_str = '\n'
        outputfile.write(output_str)
        ID = trainingset[article_id]

        if article_id % 10 == 0:
            print('Total complete articles:', article_id)

    # close output file
    outputfile.close()

trainingset, position, mentions = loadInputFile(file_path)
data_path='data/sample.data'
CRFFormatData(trainingset, position, data_path)

trainingset2, position2, mentions2 = loadInputFile("data/development_2.txt")
data_path2='data/development.data'
CRFFormatData(trainingset2, position2, data_path2)


def CRF(x_train, y_train, x_test, y_test):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    print("CRF fitting...")
    crf.fit(x_train, y_train)
    print("CRF:")
    print(crf)

    y_pred = crf.predict(x_test)
    print("y_pred:")
    print(y_pred)
    y_pred_mar = crf.predict_marginals(x_test)
    print("y_predict_marginals:")
    print(y_pred_mar)

    labels = list(crf.classes_)
    labels.remove('O')
    f1score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0])) # group B and I results
    print(flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))

    return y_pred, y_pred_mar, f1score


# load pretrained word vectors
# get a dict of tokens (key) and their pretrained word vectors (value)
# pretrained word2vec CBOW word vector: https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305ddf5522015de5479f4701b1
dim = 0
word_vecs = {}
working_sign = ['|', '/', '-', '\\']
# open pretrained word vector file
with open('data/cna.cbow.cwe_p.tar_g.512d.0.txt') as f:
    for i, line in enumerate(f):
        print("\rLoad pretrained word vectors %s"%(working_sign[i%len(working_sign)],), end="")

        tokens = line.strip().split()

        # there 2 integers in the first line: vocabulary_size, word_vector_dim
        if len(tokens) == 2:
            dim = int(tokens[1])
            continue

        word = tokens[0]
        vec = np.array([float(t) for t in tokens[1:]])
        word_vecs[word] = vec

print('\nvocabulary_size: ',len(word_vecs),' word_vector_dim: ',vec.shape)

# load `train.data` and separate into a list of labeled data of each text
# return:
#   data_list: a list of lists of tuples, storing tokens and labels (wrapped in tuple) of each text in `train.data`
#   traindata_list: a list of lists, storing training data_list splitted from data_list
#   testdata_list: a list of lists, storing testing data_list splitted from data_list
from sklearn.model_selection import train_test_split


def Dataset(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()  # .encode('utf-8').decode('utf-8-sig')
    data_list, data_list_tmp = list(), list()
    article_id_list = list()
    idx = 0
    for row in data:
        data_tuple = tuple()
        if row == '\n':
            article_id_list.append(idx)
            idx += 1
            data_list.append(data_list_tmp)
            data_list_tmp = []
        else:
            row = row.strip('\n').split(' ')
            data_tuple = (row[0], row[1])
            data_list_tmp.append(data_tuple)
    if len(data_list_tmp) != 0:
        data_list.append(data_list_tmp)

    # here we random split data into training dataset and testing dataset
    # but you should take `development data` or `test data` as testing data
    # At that time, you could just delete this line,
    # and generate data_list of `train data` and data_list of `development/test data` by this function
    traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = \
        train_test_split(data_list, article_id_list, test_size=0.33, random_state=42)

    return data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list

# look up word vectors
# turn each word into its pretrained word vector
# return a list of word vectors corresponding to each token in train.data
def Word2Vector(data_list, embedding_dict):
    embedding_list = list()

    # No Match Word (unknown word) Vector in Embedding
    unk_vector=np.random.rand(*(list(embedding_dict.values())[0].shape))

    data_list_size = len(data_list)
    for idx_list in range(data_list_size):
        print("\rWord2Vector: %d/%d"%(idx_list+1, data_list_size), end="")
        embedding_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            key = data_list[idx_list][idx_tuple][0] # token

            if key in embedding_dict:
                value = embedding_dict[key]
            else:
                value = unk_vector
            embedding_list_tmp.append(value)
        embedding_list.append(embedding_list_tmp)

    print()
    return embedding_list

# input features: pretrained word vectors of each token
# return a list of feature dicts, each feature dict corresponding to each token
def Feature(embed_list):
    feature_list = list()
    for idx_list in range(len(embed_list)):
        print("\rFeature: %d/%d"%(idx_list, len(embed_list)-1), end="")

        feature_list_tmp = list()
        for idx_tuple in range(len(embed_list[idx_list])):
            feature_dict = dict()
            for idx_vec in range(len(embed_list[idx_list][idx_tuple])):
                feature_dict['dim_' + str(idx_vec+1)] = embed_list[idx_list][idx_tuple][idx_vec]
            feature_list_tmp.append(feature_dict)
        feature_list.append(feature_list_tmp)
    print()

    return feature_list

# get the labels of each tokens in train.data
# return a list of lists of labels
def Preprocess(data_list):
    label_list = list()
    for idx_list in range(len(data_list)):
        print("\rPreprocess: %d/%d"%(idx_list, len(data_list)-1), end="")

        label_list_tmp = list()
        for idx_tuple in range(len(data_list[idx_list])):
            label_list_tmp.append(data_list[idx_list][idx_tuple][1])
        label_list.append(label_list_tmp)
    print()

    return label_list


# Training

data_list, traindata_list, testdata_list, traindata_article_id_list, testdata_article_id_list = Dataset(data_path)
traindata_list.extend(testdata_list)
traindata_article_id_list.extend(testdata_article_id_list)

data_list2, devdata_list, testdata_list2, devdata_article_id_list, testdata_article_id_list2 = Dataset(data_path2)
devdata_list.extend(testdata_list2)
devdata_article_id_list.extend(testdata_article_id_list2)

# testdata_list = traindata_list2
# testdata_article_id_list = traindata_article_id_list2

# Load Word Embedding
trainembed_list = Word2Vector(traindata_list, word_vecs)
testembed_list = Word2Vector(testdata_list, word_vecs)
devembed_list = Word2Vector(devdata_list, word_vecs)

# CRF - Train Data (Augmentation Data)
print("CRF - Train Data (Augmentation Data)")
x_train = Feature(trainembed_list)
y_train = Preprocess(traindata_list)

# CRF - Test Data (Golden Standard)
print("CRF - Test Data (Golden Standard)")
x_test = Feature(testembed_list)
y_test = Preprocess(testdata_list)

# CRF - Dev Data (Golden Standard)
# print("CRF - Dev Data (Golden Standard)")
# x_test = Feature(devembed_list)
# y_test = Preprocess(devdata_list)

y_pred, y_pred_mar, f1score = CRF(x_train, y_train, x_test, y_test)


# Output data
print("Output data...")
output = "article_id\tstart_position\tend_position\tentity_text\tentity_type\n"
y_pred_size = len(y_pred)
for test_id in range(y_pred_size):
    print("\rtest_id: %d/%d"%(test_id, y_pred_size-1), end="")

    pos = 0
    start_pos=None
    end_pos = None
    entity_text = None
    entity_type = None

    pred_id_size = len(y_pred[test_id])
    for pred_id in range(pred_id_size):
        print("\r\tpred_id: %d/%d" % (pred_id, pred_id_size - 1), end="")

        if y_pred[test_id][pred_id][0]=='B':
            start_pos=pos
            entity_type=y_pred[test_id][pred_id][2:]
        elif start_pos is not None \
                and y_pred[test_id][pred_id][0] == 'I' \
                and y_pred[test_id][pred_id+1][0] == 'O':
            end_pos=pos
            entity_text=''.join([testdata_list[test_id][position][0]
                                 for position in range(start_pos,end_pos+1)])
            line=str(testdata_article_id_list[test_id])+'\t'+str(start_pos)+'\t'+str(end_pos+1)\
                 +'\t'+entity_text+'\t'+entity_type
            output+=line+'\n'
        pos += 1
print()

output_path='output.tsv'
with open(output_path,'w',encoding='utf-8') as f:
    f.write(output)

print(output)
