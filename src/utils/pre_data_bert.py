# encoding: utf-8

import time, sys, os, codecs, random, re, math
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertTokenizer

# 情绪标签映射 (与原版保持一致)
emotion_idx = dict(zip(['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise'], range(7)))

def print_time():
    """打印当前时间"""
    print('\n----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime())))

def batch_index(length, batch_size, test=False):
    """生成批次索引，与原版保持一致"""
    index = list(range(length))
    if not test: 
        np.random.shuffle(index)
    for i in range(int((length + batch_size - 1) / batch_size)):
        ret = index[i * batch_size : (i + 1) * batch_size]
        if not test and len(ret) < batch_size: 
            break
        yield ret

def list_round(a_list):
    """列表元素四舍五入，与原版保持一致"""
    a_list = list(a_list)
    return [float('{:.4f}'.format(i)) for i in a_list]

def token_seq(text):
    """文本分词，与原版保持一致"""
    return text.split()

def load_w2v(embedding_dim, embedding_dim_pos, data_file_path, embedding_path):
    """
    加载词向量，与原版保持一致
    返回: word_idx_rev, word_idx, spe_idx_rev, spe_idx, embedding, embedding_pos
    """
    print('\nload embedding...')
    words = []
    speakers = []
    speaker_dict = {}
    
    # 读取数据文件，收集词汇和说话人
    with open(data_file_path, 'r', encoding='utf-8') as inputFile:
        while True:
            line = inputFile.readline()
            if line == '': 
                break
            line = line.strip().split()
            d_len = int(line[1])
            inputFile.readline()  # 跳过emotion-cause pairs行
            
            for i in range(d_len):
                new_line = inputFile.readline().strip().split(' | ')
                speaker, emotion, utterance = new_line[1], new_line[2], new_line[3]
                
                if speaker in speaker_dict:
                    speaker_dict[speaker] += 1
                else:
                    speaker_dict[speaker] = 1
                speakers.append(speaker)
       
                words.extend([emotion] + token_seq(utterance))

    # 构建词汇索引
    words = set(words)
    word_idx = dict((c, k + 1) for k, c in enumerate(words)) 
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words)) 

    # 构建说话人索引
    speaker_dict = sorted(speaker_dict.items(), key=lambda x: x[1], reverse=True)
    speakers = [item[0] for item in speaker_dict]
    spe_idx = dict((c, k + 1) for k, c in enumerate(speakers)) 
    spe_idx_rev = dict((k + 1, c) for k, c in enumerate(speakers))

    # 加载预训练词向量
    w2v = {}
    with open(embedding_path, 'r', encoding='utf-8') as inputFile:
        emb_cnt = int(inputFile.readline().split()[0])
        for line in inputFile.readlines():
            line = line.strip().split()
            w, ebd = line[0], line[1:]
            w2v[w] = ebd

    # 构建embedding矩阵
    embedding = [list(np.zeros(embedding_dim))]  # 第0个位置为padding
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)
    
    print('data_file: {}\nw2v_file: {}\nall_words_emb {} all_words_file: {} hit_words: {}'.format(
        data_file_path, embedding_path, emb_cnt, len(words), hit))

    # 构建位置embedding
    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)
    
    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, spe_idx_rev, spe_idx, embedding, embedding_pos

def load_embedding_from_npy(video_id_mapping_file, video_emb_file, audio_emb_file):
    """
    从npy文件加载多模态特征，与原版保持一致
    """
    def normalize(x):
        x1 = x[1:,:]
        min_x = np.min(x1, axis=0, keepdims=True)
        max_x = np.max(x1, axis=0, keepdims=True)
        x1 = (x1-min_x)/(max_x-min_x+1e-8)
        x[1:,:] = x1
        return x

    v_id_map = eval(str(np.load(video_id_mapping_file, allow_pickle=True)))
    v_emb = normalize(np.load(video_emb_file, allow_pickle=True))
    a_emb = normalize(np.load(audio_emb_file, allow_pickle=True))
    
    print('\nload video_emb_file: {}\nload audio_emb_file: {}\n'.format(video_emb_file, audio_emb_file))
    return v_id_map, v_emb, a_emb

def bert_word2id_hier(words, tokenizer, i, x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp):
    """BERT分层编码，与原版保持一致"""
    tokens_a = tokenizer.tokenize(words)
    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"] 
    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)

    start_idx = s_idx_bert_tmp[i]
    sen_len_tmp = len(input_ids)
    s_idx_bert_tmp[i+1] = start_idx + sen_len_tmp
    
    for j in range(sen_len_tmp):
        if start_idx + j < len(x_bert_tmp):
            x_bert_tmp[start_idx+j] = input_ids[j]
            x_mask_bert_tmp[start_idx+j] = 1
            x_type_bert_tmp[start_idx+j] = i % 2
        else:
            break

def bert_word2id_ind(words, max_sen_len_bert, tokenizer, i, x_bert_sen_tmp, x_mask_bert_sen_tmp):
    """BERT独立编码，与原版保持一致"""
    tokens_a, ret = tokenizer.tokenize(words), 0
    if len(tokens_a) > max_sen_len_bert - 2:
        ret += 1
        tokens_a = tokens_a[0:(max_sen_len_bert - 2)]
    tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
    
    for j in range(len(input_ids)):
        x_bert_sen_tmp[i][j] = input_ids[j]
        x_mask_bert_sen_tmp[i][j] = 1
    return ret

def cut_by_max_len(x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp, d_len, max_len=512):
    """按最大长度截断，与原版保持一致"""
    if s_idx_bert_tmp[d_len] > max_len:
        new_s_idx_bert_tmp = np.array(s_idx_bert_tmp)
        clause_max_len = max_len // d_len
        j = 0
        
        for i in range(d_len):
            start, end = s_idx_bert_tmp[i], s_idx_bert_tmp[i+1]
            if end-start <= clause_max_len:
                for k in range(start, end):
                    if j < len(x_bert_tmp) and k < len(x_bert_tmp):
                        x_bert_tmp[j] = x_bert_tmp[k]
                        x_type_bert_tmp[j] = x_type_bert_tmp[k]
                        j+=1
                    else:
                        break
                new_s_idx_bert_tmp[i+1] = new_s_idx_bert_tmp[i] + end - start
            else:
                for k in range(start, start+clause_max_len-1):
                    if j < len(x_bert_tmp) and k < len(x_bert_tmp):
                        x_bert_tmp[j] = x_bert_tmp[k]
                        x_type_bert_tmp[j] = x_type_bert_tmp[k]
                        j+=1
                    else:
                        break
                if j < len(x_bert_tmp) and end-1 < len(x_bert_tmp):
                    x_bert_tmp[j] = x_bert_tmp[end-1]
                    x_type_bert_tmp[j] = x_type_bert_tmp[end-1]
                j+=1
                new_s_idx_bert_tmp[i+1] = new_s_idx_bert_tmp[i] + clause_max_len
        
        x_bert_tmp[j:] = 0
        x_mask_bert_tmp[j:] = 0
        x_type_bert_tmp[j:] = 0
        s_idx_bert_tmp = new_s_idx_bert_tmp
    
    x_bert_tmp = x_bert_tmp[:max_len]
    x_mask_bert_tmp = x_mask_bert_tmp[:max_len]
    x_type_bert_tmp = x_type_bert_tmp[:max_len]
    return x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp

def load_data_utt_conv(input_file, tokenizer, word_idx, video_idx, spe_idx, max_doc_len, max_sen_len, 
                      max_doc_len_bert, max_sen_len_bert, model_type='', choose_emocate=''):
    """
    加载会话数据，与原版保持一致
    返回: x_bert_sen, x_mask_bert_sen, x_bert, x_mask_bert, x_type_bert, s_idx_bert, 
          x, sen_len, doc_len, speaker, x_v, y_emotion, y_cause, doc_id, y_pairs
    """
    print('\nload data_file: {}\n'.format(input_file))
    doc_id, y_emotion, y_cause = [], [], []
    x_bert_sen, x_mask_bert_sen = [], []
    x_bert, x_mask_bert, x_type_bert, s_idx_bert = [], [], [], []
    x, x_v, sen_len, doc_len, speaker, y_pairs = [], [], [], [], [], []
    num_token = []
    cut_num, cut_num_sen, cut_num_bert_sen = 0, 0, 0
    num_emo, num_emo_cause, num_pairs = 0, 0, 0
    
    with open(input_file, 'r', encoding='utf-8') as inputFile:
        while True:
            line = inputFile.readline()
            if line == '': 
                break
            line = line.strip().split()
            d_id, d_len = line[0], int(line[1])
            doc_id.append(d_id)
            doc_len.append(d_len)

            # 读取emotion-cause pairs
            pairs = eval('[' + inputFile.readline().strip() + ']')
            pair_emo, cause = [], []
            if pairs != []: 
                if len(pairs[0]) > 2:
                    pairs = [(p[0],p[1]) for p in pairs]
                    pairs = sorted(list(set(pairs)))
                pair_emo, cause = zip(*pairs)
            y_pairs.append(pairs)
            num_pairs += len(pairs)
            num_emo_cause += len(list(set(pair_emo)))

            # 初始化数组
            if choose_emocate:
                y_emotion_tmp = np.zeros((max_doc_len, 7))
            else:
                y_emotion_tmp = np.zeros((max_doc_len, 2))
            y_cause_tmp = np.zeros((max_doc_len, 2))
            
            x_tmp = np.zeros((max_doc_len, max_sen_len), dtype=np.int32)
            x_v_tmp = np.zeros(max_doc_len, dtype=np.int32)
            sen_len_tmp = np.zeros(max_doc_len, dtype=np.int32)
            spe_tmp = np.zeros(max_doc_len, dtype=np.int32)
            
            x_bert_sen_tmp = np.zeros((max_doc_len, max_sen_len_bert), dtype=np.int32)
            x_mask_bert_sen_tmp = np.zeros((max_doc_len, max_sen_len_bert), dtype=np.int32)
            
            x_bert_tmp = np.zeros(max_doc_len_bert, dtype=np.int32)
            x_mask_bert_tmp = np.zeros(max_doc_len_bert, dtype=np.int32)
            x_type_bert_tmp = np.zeros(max_doc_len_bert, dtype=np.int32)
            s_idx_bert_tmp = np.zeros(max_doc_len + 1, dtype=np.int32)

            # 处理每个话语
            for i in range(d_len):
                line = inputFile.readline().strip().split(' | ')
                speaker_name, emotion, utterance = line[1], line[2], line[3]
                
                # 设置情绪标签
                if choose_emocate:
                    if emotion_idx[emotion] > 0:
                        y_emotion_tmp[i][emotion_idx[emotion]] = 1
                        num_emo += 1
                    else:
                        y_emotion_tmp[i][0] = 1
                else:
                    if emotion_idx[emotion] > 0:
                        y_emotion_tmp[i][1] = 1
                        num_emo += 1
                    else:
                        y_emotion_tmp[i][0] = 1

                # 设置原因标签
                if i+1 in cause:
                    y_cause_tmp[i][1] = 1
                else:
                    y_cause_tmp[i][0] = 1

                # 处理文本
                words_seq = token_seq(utterance)
                sen_len_tmp[i] = min(len(words_seq), max_sen_len)
                for j, word in enumerate(words_seq):
                    if j >= max_sen_len:
                        cut_num += 1
                        break
                    if word in word_idx:
                        x_tmp[i][j] = word_idx[word]
                    else:
                        x_tmp[i][j] = word_idx.get(word, 0)  # 使用0作为未知词

                # 处理说话人
                if speaker_name in spe_idx:
                    spe_tmp[i] = spe_idx[speaker_name]

                # 处理视频特征
                video_key = 'dia{}utt{}'.format(d_id, i+1)
                if video_key in video_idx:
                    x_v_tmp[i] = video_idx[video_key]

                # BERT编码
                if model_type in ['BERTcased', 'BERTuncased']:
                    if 'BERT_sen' in tokenizer.__class__.__name__ or True:  # 简化判断
                        cut_num_bert_sen += bert_word2id_ind(utterance, max_sen_len_bert, tokenizer, i, x_bert_sen_tmp, x_mask_bert_sen_tmp)
                    bert_word2id_hier(utterance, tokenizer, i, x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp)

            # 处理BERT截断
            if model_type in ['BERTcased', 'BERTuncased']:
                x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp = cut_by_max_len(
                    x_bert_tmp, x_mask_bert_tmp, x_type_bert_tmp, s_idx_bert_tmp, d_len, max_doc_len_bert)

            # 添加到结果列表
            y_emotion.append(y_emotion_tmp)
            y_cause.append(y_cause_tmp)
            x.append(x_tmp)
            sen_len.append(sen_len_tmp)
            speaker.append(spe_tmp)
            x_v.append(x_v_tmp)
            x_bert_sen.append(x_bert_sen_tmp)
            x_mask_bert_sen.append(x_mask_bert_sen_tmp)
            x_bert.append(x_bert_tmp)
            x_mask_bert.append(x_mask_bert_tmp)
            x_type_bert.append(x_type_bert_tmp)
            s_idx_bert.append(s_idx_bert_tmp)
            num_token.append(s_idx_bert_tmp[d_len])

    # 转换为numpy数组
    arrays = [x_bert_sen, x_mask_bert_sen, x_bert, x_mask_bert, x_type_bert, s_idx_bert, 
              x, sen_len, doc_len, speaker, x_v, y_emotion, y_cause]
    x_bert_sen, x_mask_bert_sen, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, speaker, x_v, y_emotion, y_cause = map(np.array, arrays)
    
    # 打印统计信息
    converted_arrays = [x_bert_sen, x_mask_bert_sen, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, speaker, x_v, y_emotion, y_cause]
    for var_name, var_val in zip(['x_bert_sen', 'x_mask_bert_sen', 'x_bert', 'x_mask_bert', 'x_type_bert', 's_idx_bert', 'x', 'sen_len', 'doc_len', 'speaker', 'x_v', 'y_emotion', 'y_cause'], converted_arrays):
        print('{}.shape {}'.format(var_name, var_val.shape))
    
    print('num_emo: {}, num_pairs: {}, num_emo_cause: {}, cut_num: {}, cut_num_sen: {}, cut_num_bert_sen: {}'.format(
        num_emo, num_pairs, num_emo_cause, cut_num, cut_num_sen, cut_num_bert_sen))
    print('load data done!\n')

    return x_bert_sen, x_mask_bert_sen, x_bert, x_mask_bert, x_type_bert, s_idx_bert, x, sen_len, doc_len, speaker, x_v, y_emotion, y_cause, doc_id, y_pairs

# 评估指标函数 (与原版保持一致)
def cal_prf(pred_y, true_y, doc_len, average='binary'): 
    """计算precision, recall, f1"""
    pred_num, acc_num, true_num = 0, 0, 0
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            if pred_y[i][j]:
                pred_num += 1
            if true_y[i][j]:
                true_num += 1
            if pred_y[i][j] and true_y[i][j]:
                acc_num += 1
    p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
    f = 2*p*r/(p+r+1e-8)
    return p, r, f

def cal_prf_emocate(pred_y, true_y, doc_len): 
    """计算多类别情绪的precision, recall, f1"""
    conf_mat = np.zeros([7,7])
    for i in range(pred_y.shape[0]):
        for j in range(doc_len[i]):
            conf_mat[true_y[i][j]][pred_y[i][j]] += 1
    p = np.diagonal( conf_mat / np.reshape(np.sum(conf_mat, axis = 0) + 1e-8, [1,7]) )
    r = np.diagonal( conf_mat / np.reshape(np.sum(conf_mat, axis = 1) + 1e-8, [7,1]) )
    f = 2*p*r/(p+r+1e-8)
    weight = np.sum(conf_mat, axis = 1) / np.sum(conf_mat)
    w_avg_f = np.sum(f * weight)
    return np.append(f, w_avg_f)

def load_data_utt_step2(input_file, word_idx, video_idx, max_sen_len=45, choose_emocate='', pred_future_cause=1, test_bound=''):
    """
    加载step2数据，与原版保持一致
    返回: x, sen_len, distance, x_emocate, x_v, y, pair_id_all, pair_id, doc_id_list, y_pairs
    """
    print('\nload data_file: {}\n'.format(input_file))
    max_doc_len = 35
    x, sen_len, distance, x_emocate, x_v, y, y_pairs, pair_id_all, pair_id, doc_id_list = [[] for i in range(10)]
    
    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id, d_len = int(line[0]), int(line[1])
        doc_id_list.append(doc_id)
        
        pairs = eval(inputFile.readline().strip())
        if pairs != []: 
            if len(pairs[0]) > 2:
                pairs = [(p[0],p[1])  for p in pairs]
                pairs = sorted(list(set(pairs))) # If the pair contains the indexes of cause span (len(p)=4), we need to remove duplicates when taking the utterance pairs.
        if not pred_future_cause:
            pairs = [(p[0],p[1])  for p in pairs  if (p[1]-p[0])<=0]
        y_pairs.append(pairs)
        true_cause_list = sorted(list(set([p[1] for p in pairs])))

        x_tmp = np.zeros((max_doc_len, max_sen_len),dtype=np.int32)
        sen_len_tmp, y_emo_tmp, predy_emo_tmp = [np.zeros(max_doc_len,dtype=np.int32) for _ in range(3)]
        emo_list, cause_list, true_emo_list = [[] for _ in range(3)]
        
        for i in range(d_len):
            line = inputFile.readline().strip().split(' | ')
            predy_emo_tmp[i] = int(line[1].strip())

            if int(line[1].strip())>0:
                emo_list.append(i+1)
            if int(line[2].strip())>0:
                cause_list.append(i+1)
            if int(emotion_idx[line[4].strip()])>0:
                true_emo_list.append(i+1)

            y_emo_tmp[i] = emotion_idx[line[4].strip()]

            words = line[5]
            words_seq = token_seq(words)
            sen_len_tmp[i] = min(len(words_seq), max_sen_len)
            for j, word in enumerate(words_seq):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                if word in word_idx:
                    x_tmp[i][j] = word_idx[word]
                else:
                    x_tmp[i][j] = word_idx.get(word, 0)  # 使用0作为未知词

        for p in pairs:
            new_p = [doc_id, p[0], p[1], y_emo_tmp[p[0]-1]]
            pair_id_all.append(new_p)
        
        if test_bound=='EC':
            emo_list = true_emo_list
        if test_bound=='CE':
            cause_list = true_cause_list

        pair_flag = False
        for i in emo_list:
            for j in cause_list:
                if pred_future_cause:
                    pair_flag = True
                else:
                    if i>=j:
                        pair_flag = True
                    else:
                        pair_flag = False

                if pair_flag:
                    if choose_emocate:
                        pair_id_cur = [doc_id, i, j, predy_emo_tmp[i-1]]
                        if test_bound=='EC':
                            pair_id_cur = [doc_id, i, j, y_emo_tmp[i-1]]
                    else:
                        pair_id_cur = [doc_id, i, j, y_emo_tmp[i-1]]
                    pair_id.append(pair_id_cur)
                    y.append([0,1] if pair_id_cur in pair_id_all else [1,0])
                    x.append([x_tmp[i-1],x_tmp[j-1]])
                    sen_len.append([sen_len_tmp[i-1], sen_len_tmp[j-1]])
                    distance.append(j-i+100)
                    if test_bound=='EC':
                        x_emocate.append(y_emo_tmp[i-1])
                    else:
                        x_emocate.append(predy_emo_tmp[i-1])
                    
                    # 处理视频特征索引
                    x_v_i = video_idx.get('dia{}utt{}'.format(doc_id, i), 0)
                    x_v_j = video_idx.get('dia{}utt{}'.format(doc_id, j), 0)
                    x_v.append([x_v_i, x_v_j])

    x, sen_len, distance, x_emocate, x_v, y = map(np.array, [x, sen_len, distance, x_emocate, x_v, y])
    for var in ['x', 'sen_len', 'distance', 'x_emocate', 'x_v', 'y']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_pairs: {}, n_cut: {}, (y-negative, y-positive): {}'.format(len(pair_id_all), n_cut, y.sum(axis=0)))
    print('load data done!\n')
    
    return x, sen_len, distance, x_emocate, x_v, y, pair_id_all, pair_id, doc_id_list, y_pairs

def prf_2nd_step(pair_id_all, pair_id, pred_y):
    """
    计算step2的precision, recall, f1，与原版保持一致
    """
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])
    
    def cal_prf(pair_id_all, pair_id):
        acc_num, true_num, pred_num = 0, len(pair_id_all), len(pair_id)
        for p in pair_id:
            if p in pair_id_all:
                acc_num += 1
        p, r = acc_num/(pred_num+1e-8), acc_num/(true_num+1e-8)
        f1 = 2*p*r/(p+r+1e-8)
        return [p, r, f1]
    
    keep_rate = len(pair_id_filtered)/(len(pair_id)+1e-8)
    return cal_prf(pair_id_all, pair_id_filtered) + cal_prf(pair_id_all, pair_id) + [keep_rate]

def prf_2nd_step_emocate(pair_id_all, pair_id, pred_y):
    """
    计算step2情绪类别的precision, recall, f1，与原版保持一致
    """
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_filtered.append(pair_id[i])
    keep_rate = len(pair_id_filtered)/(len(pair_id)+1e-8)

    def cal_prf_emocate(pair_id_all, pair_id):
        conf_mat = np.zeros([7,7])
        for p in pair_id:
            if p in pair_id_all:
                conf_mat[p[3]][p[3]] += 1
            else:
                conf_mat[0][p[3]] += 1
        for p in pair_id_all:
            if p not in pair_id:
                conf_mat[p[3]][0] += 1
        p = np.diagonal( conf_mat / np.reshape(np.sum(conf_mat, axis = 0) + 1e-8, [1,7]) )
        r = np.diagonal( conf_mat / np.reshape(np.sum(conf_mat, axis = 1) + 1e-8, [7,1]) )
        f = 2*p*r/(p+r+1e-8)
        weight0 = np.sum(conf_mat, axis = 1)
        weight = weight0[1:] / np.sum(weight0[1:])
        w_avg_p = np.sum(p[1:] * weight)
        w_avg_r = np.sum(r[1:] * weight)
        w_avg_f = np.sum(f[1:] * weight)

        # 不考虑占比较小的disgust/fear ['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        idx = [1,4,5,6]
        weight1 = weight0[idx]
        weight = weight1 / np.sum(weight1)
        w_avg_p_part = np.sum(p[idx] * weight)
        w_avg_r_part = np.sum(r[idx] * weight)
        w_avg_f_part = np.sum(f[idx] * weight)
        
        results = list(f[1:]) + [w_avg_f, w_avg_p, w_avg_r, w_avg_f_part, w_avg_p_part, w_avg_r_part]  
        # results = list(f[1:]) + [w_avg_p, w_avg_r, w_avg_f, w_avg_p_part, w_avg_r_part, w_avg_f_part]
        return results
    
    return cal_prf_emocate(pair_id_all, pair_id_filtered) + cal_prf_emocate(pair_id_all, pair_id) + [keep_rate]