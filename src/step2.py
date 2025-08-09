# encoding: utf-8

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
print('\ntorch: {}\ntorch.cuda.is_available: {}\n'.format(torch.__version__, torch.cuda.is_available()))
import numpy as np
import sys, os, time, codecs, pdb

sys.path.append('./src/utils')
sys.path.append('./utils')
try:
    from tf_funcs import *
    from pre_data_bert import *
except ImportError:
    from src.utils.tf_funcs import *
    from src.utils.pre_data_bert import *

# 参数配置类 (对应原版的FLAGS)
class Config:
    def __init__(self):
        # 获取脚本所在目录的绝对路径
        import os
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
        ## embedding parameters ##
        self.w2v_file = os.path.join(script_dir, 'data/ECF_glove_300.txt')
        self.path = os.path.join(script_dir, 'data/')
        self.video_emb_file = os.path.join(script_dir, 'data/video_embedding_4096.npy')
        self.audio_emb_file = os.path.join(script_dir, 'data/audio_embedding_6373.npy')
        self.video_idx_file = os.path.join(script_dir, 'data/video_id_mapping.npy')
        self.embedding_dim = 300
        self.embedding_dim_pos = 50
        ## input struct ##
        self.max_sen_len = 35
        self.pred_future_cause = 1
        ## model struct ##
        self.choose_emocate = ''
        self.emocate_eval = 6
        self.use_x_v = 'use'
        self.use_x_a = 'use'
        self.n_hidden = 100
        self.n_class = 2
        # >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
        self.batch_size = 200
        self.learning_rate = 0.005
        self.keep_prob1 = 0.5
        self.keep_prob2 = 1.0
        self.l2_reg = 1e-5
        self.keep_prob_v = 0.5
        self.keep_prob_a = 0.5
        self.end_run = 21
        self.training_iter = 12

        self.log_path = os.path.join(script_dir, 'log')
        self.scope = 'TEMP'
        self.log_file_name = 'step2.log'
        self.save_pair = 'yes'
        self.conv_file_dir = 'conv/'

FLAGS = Config()

def print_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>MODEL INFO:')
    print('choose_emocate: {}\nemocate_eval: {}\nvideo_emb_file: {}\naudio_emb_file: {}\nuse_x_v: {}\nuse_x_a: {}\n\n'.format(
        FLAGS.choose_emocate, FLAGS.emocate_eval, FLAGS.video_emb_file, FLAGS.audio_emb_file, FLAGS.use_x_v, FLAGS.use_x_a))

    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:')
    print('path: {}\nbatch: {}\nlr: {}\nkb1: {}\nkb2: {}\nl2_reg: {}\nkeep_prob_v: {}\nkeep_prob_a: {}\ntraining_iter: {}\nend_run: {}\npred_future_cause: {}\nconv_file_dir: {}\n\n'.format(
        FLAGS.path, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg, FLAGS.keep_prob_v, FLAGS.keep_prob_a, FLAGS.training_iter, FLAGS.end_run, FLAGS.pred_future_cause, FLAGS.conv_file_dir))

class MECPE_Step2_Model(nn.Module):
    """MECPE Step2 PyTorch模型"""
    def __init__(self, word_embedding, pos_embedding, video_embedding, audio_embedding, config):
        super(MECPE_Step2_Model, self).__init__()
        self.config = config
        
        # 从实际嵌入矩阵获取维度
        actual_embedding_dim = word_embedding.shape[1]
        actual_pos_embedding_dim = pos_embedding.shape[1]
        
        # Embedding层
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=False)
        self.pos_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pos_embedding), freeze=False)
        self.video_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(video_embedding), freeze=False)
        self.audio_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(audio_embedding), freeze=False)
        
        # LSTM层 - 使用实际的embedding维度
        self.bilstm = BiLSTM(actual_embedding_dim, config.n_hidden)
        
        # 注意力层
        h2 = 2 * config.n_hidden
        self.attention_layer = AttentionLayer(h2, h2)
        
        # 多模态特征变换
        if config.use_x_v:
            self.video_transform = nn.Linear(video_embedding.shape[1], h2)
        if config.use_x_a:
            self.audio_transform = nn.Linear(audio_embedding.shape[1], h2)
        
        # 最终分类层
        dim_s = h2
        if config.use_x_v:
            dim_s += h2
        if config.use_x_a:
            dim_s += h2
        
        final_dim = 2 * dim_s + actual_pos_embedding_dim  # 2个句子的特征 + 距离嵌入
        if config.choose_emocate:
            final_dim += actual_pos_embedding_dim  # + 情绪类别嵌入
            
        self.classifier = nn.Linear(final_dim, config.n_class)
        
        # Dropout
        self.dropout1 = nn.Dropout(1 - config.keep_prob1)
        self.dropout2 = nn.Dropout(1 - config.keep_prob2)
        self.dropout_v = nn.Dropout(1 - config.keep_prob_v)
        self.dropout_a = nn.Dropout(1 - config.keep_prob_a)
        
    def forward(self, x, sen_len, distance, x_emocate, x_v, is_training=True):
        """
        x: [batch_size, 2, max_sen_len] - 两个句子的词索引
        sen_len: [batch_size, 2] - 两个句子的长度
        distance: [batch_size] - 句子之间的距离
        x_emocate: [batch_size] - 情绪类别
        x_v: [batch_size, 2] - 两个句子的视频/音频索引
        """
        batch_size = x.size(0)
        
        # 词嵌入
        inputs = self.word_embedding(x)  # [batch_size, 2, max_sen_len, embedding_dim]
        batch_size, _, max_sen_len, embedding_dim = inputs.shape
        inputs = inputs.view(-1, max_sen_len, embedding_dim)  # [batch_size*2, max_sen_len, embedding_dim]
        if is_training:
            inputs = self.dropout1(inputs)
        
        # 句子长度处理
        sen_len_flat = sen_len.view(-1)  # [batch_size*2]
        
        # BiLSTM编码
        lstm_out = self.bilstm(inputs, sen_len_flat)  # [batch_size*2, max_sen_len, n_hidden*2]
        
        # 注意力机制
        h2 = 2 * self.config.n_hidden
        s = self.attention_layer(lstm_out, sen_len_flat)  # [batch_size*2, n_hidden*2]
        s = s.view(batch_size, 2, h2)  # [batch_size, 2, n_hidden*2]
        
        # 多模态特征
        if self.config.use_x_v:
            x_v_emb = self.video_embedding(x_v)  # [batch_size, 2, video_dim]
            if is_training:
                x_v_emb = self.dropout_v(x_v_emb)
            x_v_trans = F.relu(layer_normalize(self.video_transform(x_v_emb)))  # [batch_size, 2, h2]
            s = torch.cat([s, x_v_trans], dim=2)
            
        if self.config.use_x_a:
            x_a_emb = self.audio_embedding(x_v)  # [batch_size, 2, audio_dim] (注意：x_v同时包含视频和音频索引)
            if is_training:
                x_a_emb = self.dropout_a(x_a_emb)
            x_a_trans = F.relu(layer_normalize(self.audio_transform(x_a_emb)))  # [batch_size, 2, h2]
            s = torch.cat([s, x_a_trans], dim=2)
        
        # 特征拼接
        dim_s = s.size(-1)
        s = s.view(batch_size, 2 * dim_s)  # [batch_size, 2*dim_s]
        
        # 距离嵌入
        dis_emb = self.pos_embedding(distance)  # [batch_size, embedding_dim_pos]
        s = torch.cat([s, dis_emb], dim=1)
        
        # 情绪类别嵌入
        if self.config.choose_emocate:
            x_emocate_emb = self.pos_embedding(x_emocate)  # [batch_size, embedding_dim_pos]
            s = torch.cat([s, x_emocate_emb], dim=1)
        
        # Dropout和分类
        if is_training:
            s = self.dropout2(s)
        
        pred_pair = F.softmax(self.classifier(s), dim=1)  # [batch_size, n_class]
        
        return pred_pair

def build_model(embeddings, device='cpu'):
    """构建step2模型"""
    word_embedding, pos_embedding, video_embedding, audio_embedding = embeddings
    model = MECPE_Step2_Model(word_embedding, pos_embedding, video_embedding, audio_embedding, FLAGS)
    return model.to(device)

class MECPEStep2Dataset:
    """MECPE Step2数据集类"""
    def __init__(self, data_file_name, word_idx, video_idx):
        x, sen_len, distance, x_emocate, x_v, y, pair_id_all, pair_id, doc_id_list, y_pairs = load_data_utt_step2(
            data_file_name, word_idx, video_idx, FLAGS.max_sen_len, FLAGS.choose_emocate, FLAGS.pred_future_cause)

        self.x = torch.LongTensor(x)
        self.sen_len = torch.LongTensor(sen_len) 
        self.distance = torch.LongTensor(distance)
        self.x_emocate = torch.LongTensor(x_emocate)
        self.x_v = torch.LongTensor(x_v)
        self.y = torch.FloatTensor(y)
        
        self.pair_id_all = pair_id_all
        self.pair_id = pair_id
        self.doc_id_list = doc_id_list
        self.y_pairs = y_pairs
        
        self.all = [self.x, self.sen_len, self.distance, self.x_emocate, self.x_v, self.y]

def get_batch_data(dataset, is_training, batch_size):
    """获取批量数据"""
    test = bool(1 - is_training)
    for index in batch_index(len(dataset.x), batch_size, test):
        feed_list = [data[index] for data in dataset.all]
        yield feed_list, index

from collections import defaultdict
def create_dict(pair_list, choose_emocate):
    """创建配对字典"""
    emotion_idx_rev = dict(zip(range(7), ['neutral','anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']))
    pair_dict = defaultdict(list)
    for x in pair_list:
        if choose_emocate:
            tmp = x[1:3] + [emotion_idx_rev[x[3]]]
            pair_dict[x[0]].append(tmp)
        else:
            pair_dict[x[0]].append(x[1:-1])
    return pair_dict

def write_data_all(input_file_name, output_file_name, dataset, pred_y, tr_batch_index, choose_emocate=''):
    """写入预测结果到文件"""
    pair_id_all, pair_id, doc_id_list, y_pairs = dataset.pair_id_all, dataset.pair_id, dataset.doc_id_list, dataset.y_pairs
    if tr_batch_index:
        pair_id = np.array(pair_id)[tr_batch_index]
    
    print('pair_id: {}  pred_y: {}'.format(len(pair_id), len(pred_y)))
    pair_id_filtered = []
    for i in range(len(pair_id)):
        if pred_y[i]:
            pair_id_tmp = list(pair_id[i])
            pair_id_filtered.append(pair_id_tmp)

    pair_id_all_dict = create_dict(pair_id_all, choose_emocate)
    pair_id_filtered_dict = create_dict(pair_id_filtered, choose_emocate)

    fo = open(output_file_name, 'w', encoding='utf8')
    inputFile = open(input_file_name, 'r', encoding='utf8')
    while True:
        line = inputFile.readline()
        fo.write(line)
        if line == '': 
            break
        line = line.strip().split()
        doc_id, d_len = int(line[0]), int(line[1])
        line = inputFile.readline()
        fo.write(str(pair_id_all_dict[doc_id])+'\n')
        if doc_id in pair_id_filtered_dict:
            fo.write(str(pair_id_filtered_dict[doc_id])+'\n')
        else:
            fo.write('\n')
        for i in range(d_len):
            line = inputFile.readline()
            fo.write(line)
    print('write {} done'.format(output_file_name))

def run():
    """主运行函数"""
    if 'emocate' in FLAGS.scope:
        FLAGS.choose_emocate = 'use'
    if not os.path.exists(FLAGS.log_path):
        os.makedirs(FLAGS.log_path)
    save_dir = '{}/{}/'.format(FLAGS.log_path, FLAGS.scope)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dev_eval_list, test_eval_list = [], []
    max_epoch_list = []
    cur_run = 1
    
    while True:
        if cur_run == FLAGS.end_run: 
            break

        print_time()
        print('\n############# run {} begin ###############'.format(cur_run))

        # 加载数据和嵌入
        word_idx_rev, word_idx, _, _, word_embedding, pos_embedding = load_w2v(
            FLAGS.embedding_dim, FLAGS.embedding_dim_pos, FLAGS.path+'all_data_pair.txt', FLAGS.w2v_file)
        video_idx, video_embedding, audio_embedding = load_embedding_from_npy(
            FLAGS.video_idx_file, FLAGS.video_emb_file, FLAGS.audio_emb_file)

        # 数据文件路径
        train_file_name = 'run{}_train.txt'.format(cur_run)
        dev_file_name = 'run{}_dev.txt'.format(cur_run)
        test_file_name = 'run{}_test.txt'.format(cur_run)
        if os.path.exists(save_dir+'run1_train.txt'):
            save_dir1 = save_dir
        else:
            save_dir1 = save_dir + FLAGS.conv_file_dir

        # 加载数据集
        train_data = MECPEStep2Dataset(save_dir1+train_file_name, word_idx, video_idx) 
        dev_data = MECPEStep2Dataset(save_dir1+dev_file_name, word_idx, video_idx)
        test_data = MECPEStep2Dataset(save_dir1+test_file_name, word_idx, video_idx)
        print('train docs: {}  dev docs: {}  test docs: {}'.format(len(train_data.x), len(dev_data.x), len(test_data.x)))

        embeddings = [word_embedding, pos_embedding, video_embedding, audio_embedding]

        print('\nbuild model...')
        model = build_model(embeddings, device)
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        print('build model done!\n')
        
        # 训练代码块
        print_info()
        
        max_f1 = -1.
        max_epoch_index = 0
        
        for i in range(FLAGS.training_iter):
            start_time, step = time.time(), 1
            model.train()
            
            # 训练
            tr_predy_tofile0, tr_batch_index_list = [], []
            for train_batch, batch_index in get_batch_data(train_data, is_training=1, batch_size=FLAGS.batch_size):
                x, sen_len, distance, x_emocate, x_v, y = [t.to(device) for t in train_batch]
                
                optimizer.zero_grad()
                pred_pair = model(x, sen_len, distance, x_emocate, x_v, is_training=True)
                
                # 计算损失 (包含L2正则化)
                loss = criterion(pred_pair, y.argmax(dim=1))
                l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + FLAGS.l2_reg * l2_reg
                
                loss.backward()
                optimizer.step()
                
                # 计算准确率
                pred_y = pred_pair.argmax(dim=1)
                true_y = y.argmax(dim=1)
                acc = (pred_y == true_y).float().mean()
                
                tr_predy_tofile0.extend(pred_y.cpu().numpy().tolist())
                tr_batch_index_list.extend(batch_index)
                print('step {}: train loss {:.4f} acc {:.4f}'.format(step, loss.item(), acc.item()))
                step = step + 1

            def evaluate(test_data, is_dev=True):
                """评估函数 - 使用新的结果收集方法"""
                model.eval()
                
                # 收集所有批次的结果
                all_losses = []
                all_pred_y = []
                all_true_y = []
                
                with torch.no_grad():
                    for batch_data, batch_index in get_batch_data(test_data, is_training=0, batch_size=FLAGS.batch_size):
                        x, sen_len, distance, x_emocate, x_v, y = [t.to(device) for t in batch_data]
                        
                        pred_pair = model(x, sen_len, distance, x_emocate, x_v, is_training=False)
                        loss = criterion(pred_pair, y.argmax(dim=1))
                        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
                        loss = loss + FLAGS.l2_reg * l2_reg
                        
                        pred_y_batch = pred_pair.argmax(dim=1).cpu().numpy()
                        true_y_batch = y.argmax(dim=1).cpu().numpy()
                        
                        all_losses.append(loss.item())
                        all_pred_y.append(pred_y_batch)
                        all_true_y.append(true_y_batch)
                
                # 使用 np.concatenate 拼接完整数组
                avg_loss = np.mean(all_losses)
                complete_pred_y = np.concatenate(all_pred_y, axis=0)
                complete_true_y = np.concatenate(all_true_y, axis=0)
                
                # 对完整数组调用评价函数
                if FLAGS.choose_emocate:
                    eval_result = prf_2nd_step_emocate(test_data.pair_id_all, test_data.pair_id, complete_pred_y)
                else:
                    eval_result = prf_2nd_step(test_data.pair_id_all, test_data.pair_id, complete_pred_y)
                    
                return avg_loss, eval_result, complete_pred_y

            dev_loss, dev_eval, de_pred_y = evaluate(dev_data, is_dev=True)
            test_loss, test_eval, te_pred_y = evaluate(test_data, is_dev=False)

            dev_eval, test_eval = map(lambda x: np.array(x), [dev_eval, test_eval])
            print('\nepoch {}: cost time {:.1f} s  dev_loss {:.4f}  test_loss {:.4f}\n'.format(i, time.time()-start_time, dev_loss, test_loss))
            
            # 根据评估指标更新最佳结果
            if FLAGS.choose_emocate:
                if FLAGS.emocate_eval == 4:
                    if dev_eval[11] > max_f1:
                        max_f1 = dev_eval[11]
                        max_epoch_index = i+1
                        max_dev_eval = dev_eval
                        max_test_eval = test_eval
                        tr_predy_tofile = tr_predy_tofile0
                        de_predy_tofile = de_pred_y
                        te_predy_tofile = te_pred_y
                else:
                    if dev_eval[8] > max_f1:
                        max_f1 = dev_eval[8]
                        max_epoch_index = i+1
                        max_dev_eval = dev_eval
                        max_test_eval = test_eval
                        tr_predy_tofile = tr_predy_tofile0
                        de_predy_tofile = de_pred_y
                        te_predy_tofile = te_pred_y
                print('dev_eval \n{}\n{} \nmax_dev_eval \n{}\n{}'.format(dev_eval[:6], dev_eval[6:], max_dev_eval[:6], max_dev_eval[6:]))
                print('test_eval \n{}\n{} \nmax_test_eval \n{}\n{}\n\n'.format(test_eval[:6], test_eval[6:], max_test_eval[:6], max_test_eval[6:]))
            else:
                if dev_eval[2] > max_f1:
                    max_f1 = dev_eval[2]
                    max_epoch_index = i+1
                    max_dev_eval = dev_eval
                    max_test_eval = test_eval
                    tr_predy_tofile = tr_predy_tofile0
                    de_predy_tofile = de_pred_y
                    te_predy_tofile = te_pred_y
                print('dev_eval: {}\nmax_dev_eval: {}\n'.format(dev_eval, max_dev_eval))
                print('test_eval: {}\nmax_test_eval: {}\n\n'.format(test_eval, max_test_eval))
                
        print('Optimization Finished!\n')
        print('############# run {} end ###############\n'.format(cur_run))
        
        if max_f1 > 0.0:
            if FLAGS.save_pair:
                save_pair_path = save_dir + FLAGS.log_file_name.replace('.log', '_pair/')
                if not os.path.exists(save_pair_path):
                    os.makedirs(save_pair_path)
                write_data_all(save_dir1+train_file_name, save_pair_path+train_file_name, train_data, tr_predy_tofile, tr_batch_index_list, FLAGS.choose_emocate)
                write_data_all(save_dir1+dev_file_name, save_pair_path+dev_file_name, dev_data, de_predy_tofile, [], FLAGS.choose_emocate)
                write_data_all(save_dir1+test_file_name, save_pair_path+test_file_name, test_data, te_predy_tofile, [], FLAGS.choose_emocate)

        dev_eval_list.append(max_dev_eval)
        test_eval_list.append(max_test_eval)
        max_epoch_list.append(max_epoch_index)

        print('\n--------------- previous {} runs Avg -----------------\n'.format(cur_run))
        dev_eval_list_, test_eval_list_ = map(lambda x: np.around(np.array(x), decimals=4), [dev_eval_list, test_eval_list])
        print('\ndev_eval_list: \n{}\nAvg: {}\nStd: {}\n\n'.format(dev_eval_list_, list_round(dev_eval_list_.mean(axis=0)), dev_eval_list_.std(axis=0)))
        print('\ntest_eval_list: \n{}\nAvg: {}\nStd: {}\n\n'.format(test_eval_list_, list_round(test_eval_list_.mean(axis=0)), test_eval_list_.std(axis=0)))
        print('max_epoch:\n{}  {}  {}\n\n'.format(max_epoch_list, max(max_epoch_list), np.mean(max_epoch_list)))
        print('-----------------------------------------------------')
        
        cur_run = cur_run + 1
        
    print_time()

def main():
    run()

if __name__ == '__main__':
    main()