import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='GPU_NUMBER'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch.nn as nn
from nltk.translate.meteor_score import meteor_score
from nltk.translate import meteor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
import os
import pdb
import spacy
import pandas as pd
import json
import tqdm
import datetime
from tqdm import tqdm
from collections import Counter
from nltk import word_tokenize
import random
import math
import pdb
from rlutils import collect_samples, ppo_step, generate_n_candidates, convert_sentences_to_strings, expand_inputs_for_N_candidates
from torch.utils.data import DataLoader, Dataset
from loss import SequenceCrossEntropyLoss
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from ppo import PPOMemory
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelWithLMHead, AutoModelForCausalLM, AutoTokenizer


import warnings
warnings.filterwarnings("ignore")

model = SentenceTransformer('bert-base-nli-mean-tokens')

class Trainer():

    def __init__(self,
                 modelname,
                 train_csvfile,
                 val_csvfile,
                 n_epochs,
                 print_every,
                 learning_rate,
                 epsilon,
                 human_reward,
                 average_sent_loss,
                 device,
                 num_candidates,
                 max_candidate_length,
                 top_p,
                 warmup_steps,
                 pad_token_id,
                 evaluate_every,
                 use_persona,
                 use_strategy,
                 use_stage_intent_consistency,
                 use_diversity,
                 use_contextual_consistency,
                 use_fluency,
                 mini_batch,
                 temperature,
                 use_recent_past,
                 recompute_log_prob,
                 gamma1,
                 gamma2,
                 gamma3,
                 gamma4,
                 gamma5,
                 gamma6,
                 train_single_model=False,
                 single_model_to_train=None,
                 loadModel=False,
                 batch_size=None,
                 loadFilenameA=None,
                 loadFilenameB=None,
                 seedvalue=10):

        self.seedvalue = seedvalue
        self.train_single_model = train_single_model
        self.single_model_to_train = single_model_to_train
        self.nlp = spacy.load("en_core_web_sm")
        self.human_reward = human_reward
        self.seed(seedvalue)
        self.use_recent_past = use_recent_past
        self.temperature=temperature
        self.use_persona = use_persona
        self.use_strategy = use_strategy
        self.use_stage_intent_consistency = use_stage_intent_consistency
        self.use_diversity = use_diversity
        self.use_contextual_consistency = use_contextual_consistency
        self.use_fluency = use_fluency
        self.average_sent_loss = average_sent_loss
        self.mini_batch = mini_batch
        self.evaluate_every = evaluate_every
        self.train_csvfile = train_csvfile
        self.val_csvfile = val_csvfile
        self.modelname = modelname
        self.n_epochs = n_epochs
        self.print_every = print_every
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        
        self.recompute_log_prob = recompute_log_prob
        self.num_candidates = num_candidates
        self.pad_token_id = pad_token_id
        self.max_candidate_length = max_candidate_length
        
        
        self.top_p = top_p
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size

        self.device = device
        
        self.loadModel = loadModel
        self.loadFilenameA = loadFilenameA
        self.loadFilenameB = loadFilenameB
        self.make_model_save_dir()
        self.make_stats_dir()
        

        self.getDataset()
        
        self.initialize_models()
        self.configure_optimizer()
        
        self.buffer_memory = PPOMemory()
        
        self.saveModelConfig()
        self.criterion = SequenceCrossEntropyLoss()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.gamma3 = gamma3
        self.gamma4 = gamma4
        self.gamma5 = gamma5
        self.gamma6 = gamma6
        


    def saveModelConfig(self):
        if self.train_single_model:
            config_model_train = self.single_model_to_train
            print('Training Only :', self.single_model_to_train)
        else:
            config_model_train = 'Both Models being Trained.'
            print('Both Models being Trained.')
        config = {'Basic Info': [datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S")],
                  'NOTES': 'GPT2-MEDIUM',
                  'modelname': self.modelname,
                  'Training only one Model': self.train_single_model,
                  'Training Models': config_model_train,
                  'device': 'cuda',
                  'use_persona':self.use_persona,
                  'use_strategy': self.use_strategy,
                  'use_stage_intent_consistency':self.use_stage_intent_consistency,
                  'use_diversity': self.use_diversity,
                  'use_contextual_consistency':self.use_contextual_consistency,
                  'use_fluency':self.use_fluency,
                  'modelLoaded': self.loadFilenameA,
                  'human_reward': self.human_reward,
                  'average_sent_loss' : self.average_sent_loss,
                  'n_epochs': self.n_epochs,
                  'use_recent_past': self.use_recent_past,
                  'temperature': self.temperature,
                  'learning_rate': self.learning_rate,
                  'epsilon': self.epsilon,
                  'num_candidates': self.num_candidates,
                  'pad_token_id': self.pad_token_id,
                  'max_candidate_length': self.max_candidate_length,
                  'recompute_log_prob': self.recompute_log_prob,
                  'evaluate_every': self.evaluate_every,
                  'top_p': self.top_p,
                  'warmup_steps': self.warmup_steps,
                  'batch_size':self.batch_size,
                  'seed': self.seedvalue}
        configfilename = os.path.join(self.savefolder, self.modelname, 'config')
        if not os.path.exists(configfilename):
            os.makedirs(configfilename)
        configfilename = configfilename + '/config' + '_' + self.modelname + '.json'
        with open(configfilename, 'w') as f:
            json.dump(config, f)
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
    def seed(self,seed=10):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def extract_data(self, csvfile):
        print('Reading ',csvfile)
        df_dialogs = pd.read_csv(csvfile)
        df_dialogs['negotiation_strategy'] = df_dialogs['negotiation_strategy'].fillna('NA')
        df_dialogs['negotiation_stage'] = df_dialogs['negotiation_stage'].fillna('NA')
        data = {}
        next_utter={}
        next_strategy = {}
        next_intent = {}
        next_stage = {}
        next_persona = {}
        for i in tqdm(range(len(df_dialogs))):
            line = df_dialogs.iloc[i]
            data_line = []
            
            
            if line["id"] not in data:
                data[line["id"]] = []
                next_utter[line["id"]] = []
                next_strategy[line["id"]] = []
                next_intent[line["id"]] = []
                next_stage[line["id"]] = []
                next_persona[line["id"]] = []

            if line["speaker"] == 0:
                line_next = df_dialogs.iloc[i+1]
                next_utter[line["id"]].append(line_next["utterance"])
                next_strategy[line["id"]].append(line_next['negotiation_strategy'])  
                next_intent[line["id"]].append(line_next['intent'])
                next_stage[line["id"]].append(line_next['negotiation_stage']) 
                next_persona[line["id"]].append(line_next['aspects'])                
                text = 'A:'+"tourist to agent: '" + line["utterance"].strip() + "||"
                text = text + line['negotiation_strategy'].strip() + "||" + line['intent'].strip() + "||" + line['negotiation_stage'].strip() + "||" + line['aspects'].strip() + " "
                
                data[line["id"]].append(text)
                
            else:
                if i <= len(df_dialogs)-2:
                    line_next = df_dialogs.iloc[i+1]
                    next_utter[line["id"]].append(line_next["utterance"])
                    next_strategy[line["id"]].append(line_next['negotiation_strategy'])
                    next_intent[line["id"]].append(line_next['intent'])
                    next_stage[line["id"]].append(line_next['negotiation_stage'])
                    next_persona[line["id"]].append(line_next['aspects'])         
                else:
                    next_utter[line["id"]].append("<|end|>")
                    next_strategy[line["id"]].append("<|end|>")
                    next_intent[line["id"]].append("<|end|>")
                    next_stage[line["id"]].append("<|end|>")
                    next_persona[line["id"]].append("<|end|>")
                
                print(line['negotiation_strategy'])
                print(line['intent'].strip())
                print(line['negotiation_stage'].strip())
                print(str(line['aspects']).strip())
                text = "B:agent to tourist: '" +line["utterance"].strip() + "||"
                text = text + str(line['negotiation_strategy']).strip()  + "||" + line['intent'].strip() + "||" + str(line['negotiation_stage']).strip() + "||" + str(line['aspects']).strip() + " "
                data[line["id"]].append(text)
        
        return data, next_strategy, next_utter, next_intent, next_stage, next_persona


        
    def utteranceToConversation(self, csvfile, data, persona, path, topic):
        df = pd.read_csv(self.csvfile)
        values=df['conv_id'].unique().tolist()
        conv_ids = df['conv_id'].tolist()

        dataset = []
        conversation = []
        personaset = []
        persona_conversation = []
        pathset = []
        path_conversation = []
        topicset = []
        topic_conversation = []
        for conv in values:
            for i in range(0, df.shape[0]):
                if(conv_ids[i]==conv):
                    conversation.append(data[i])
                    persona_conversation.append(persona[i])
                    path_conversation.append(path[i])
                    topic_conversation.append(topic[i])
                else:
                    continue
            dataset.append(conversation)
            personaset.append(persona_conversation)
            pathset.append(path_conversation)
            topicset.append(topic_conversation)
            conversation = []
            persona_conversation = []
            path_conversation = []
            topic_conversation = []
        return dataset, personaset, pathset, topicset

  
          
    def convertDicttoList(self, data: dict):
        return list(data.values())

    def random_split_data(self, data, next_strategy, next_utter):
        indices = np.arange(len(data))
        np.random.shuffle(indices)


        train_data = [data[idx] for idx in indices[:800]] # XXXX: nuber of dialogues in train dataset
        val_data = [data[idx] for idx in indices[800:]]

        train_agent_strategy = [next_strategy[idx] for idx in indices[:800]]
        val_agent_strategy = [next_strategy[idx] for idx in indices[800:]]
        train_agent_utter = [next_utter[idx] for idx in indices[:800]]
        val_agent_utter = [next_utter[idx] for idx in indices[800:]]
        
        return train_data, val_data, train_agent_strategy, val_agent_strategy, train_agent_utter, val_agent_utter


    def getDataset(self):
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        train_data, train_agent_strategy, train_agent_utter, train_user_intent, train_agent_stage, train_user_persona = self.extract_data(self.train_csvfile)
        val_data, val_agent_strategy, val_agent_utter, val_user_intent, val_agent_stage, val_user_persona = self.extract_data(self.val_csvfile)

        
        traindata_ = NegotiationDataset(train_data,
                                     train_agent_strategy,
                                     train_agent_utter,
                                     train_user_intent,
                                     train_agent_stage,
                                     train_user_persona,
                                     self.tokenizer)

        print(len(traindata_))
        
        
        self.turn_ending = traindata_.turn_ending
        
        
        valdata_ = NegotiationDataset(val_data,
                                     val_agent_strategy,
                                     val_agent_utter,
                                     val_user_intent,
                                     val_agent_stage,
                                     val_user_persona,
                                     self.tokenizer)
        print(len(valdata_))
        

        self.train_dataloader = DataLoader(dataset=traindata_,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           collate_fn=traindata_.collate)
        
        self.val_dataloader = DataLoader(dataset=valdata_,
                                         shuffle=False,
                                         batch_size=self.batch_size,
                                         collate_fn=valdata_.collate)


    def initialize_models(self):
        if not self.train_single_model:
            self.model_A = AutoModelForCausalLM.from_pretrained("gpt2")
            self.model_A.to(device)
            
            self.model_B = AutoModelForCausalLM.from_pretrained("gpt2")
            self.model_B.to(device)
            self.model_A_ref = AutoModelForCausalLM.from_pretrained("gpt2")
            self.model_A_ref.to(device)
            self.model_B_ref = AutoModelForCausalLM.from_pretrained("gpt2")
            
            self.model_B_ref.to(device)
        else:
            if self.single_model_to_train == 'agent':
                self.model_A = AutoModelWithLMHead.from_pretrained("gpt2")
                self.model_A.to(device)
                self.model_A_ref = AutoModelWithLMHead.from_pretrained("gpt2")
                self.model_A_ref.to(device)
            else:
                self._model_B = AutoModelWithLMHead.from_pretrained("gpt2")
                self.model_B.to(device)
                self.model_B_ref = AutoModelWithLMHead.from_pretrained("gpt2")
                self.model_B_ref.to(device)

        if self.loadModel:
            if self.loadFilenameA:
                model_A_state_dict = torch.load(self.loadFilenameA, map_location=self.device)
                model_B_state_dict = torch.load(self.loadFilenameB, map_location=self.device)
                
                if not self.train_single_model:
                    self.model_A.load_state_dict(model_A_state_dict)
                    self.model_A_ref.load_state_dict(model_A_state_dict)
                    self.model_B.load_state_dict(model_B_state_dict)
                    self.model_B_ref.load_state_dict(model_B_state_dict)
                    self.model_A = self.model_A.to(self.device)
                    self.model_A_ref = self.model_A_ref.to(self.device)
                    self.model_B = self.model_B.to(self.device)
                    self.model_B_ref = self.model_B_ref.to(self.device)
                    self.model_A.train()
                    self.model_B.train()
                    self.model_A_ref.eval()
                    self.model_B_ref.eval()
                else:
                    if self.single_model_to_train == 'agent':
                        self.model_A.load_state_dict(model_A_state_dict)
                        self.model_A_ref.load_state_dict(model_A_state_dict)
                        self.model_A = self.model_A.to(self.device)
                        self.model_A_ref = self.model_A_ref.to(self.device)
                        self.model_A.train()
                        self.model_A_ref.eval()
                        self.model_B = None
                        self.model_B_ref = None
                    else:
                        self.model_B.load_state_dict(model_B_state_dict)
                        self.model_B_ref.load_state_dict(model_B_state_dict)
                        self.model_B = self.model_B.to(self.device)
                        self.model_B_ref = self.model_B_ref.to(self.device)
                        self.model_B.train()
                        self.model_B_ref.eval()
                        self.model_A = None
                        self.model_A_ref = None
                print('\n')
                print("Models loaded from file ", self.loadFilenameA)
            else:
                print('Models not loaded since directory not provided.')
        print(f"Models Initalized!")
        print('\n')


    def configure_optimizer(self):
        
        self.num_train_optimization_steps = self.n_epochs * 3000 # // self.batch_size  ### Hardcoded

        if not self.train_single_model:
            param_optimizer = list(self.model_A.named_parameters()) + list(self.model_B.named_parameters())
        else:
            if self.single_model_to_train == 'agent':
                param_optimizer = list(self.model_A.named_parameters())
        no_decay = ['bias', 'ln', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optimizer = AdamW(optimizer_grouped_parameters,
                                           lr=self.learning_rate,
                                           eps=1e-06)

        #self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
        #                                                 num_warmup_steps=self.warmup_steps,
        #                                                 num_training_steps=self.num_train_optimization_steps)

        '''self.scheduler = WarmupLinearSchedule(self.optimizer,
                                                 warmup_steps=self.warmup_steps,
                                                 t_total=self.num_train_optimization_steps)'''


    def get_candidate_lengths(self, candidates):



        avg_iter_length = []
        
        for i in candidates:
            candidate_sentence = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            avg_iter_length.append(len(candidate_sentence.split()))
        return avg_iter_length

    def label2idx(self,role_id,pred_strategy):
        tourist_strategy = {'na':-1,'problem-solving':0, 'strategic-proposal':1 , 'firm-pricing':2, 'definitive-decision-making':3,'collaborative-proposal':4 ,'flexible-pricing':5, 'co-operative-decision-making':6, 'no-strategy':7}
        agent_strategy = {'na':-1,'problem-solving':0, 'strategic-proposal':1 , 'firm-pricing':2, 'definitive-decision-making':3,'collaborative-proposal':4 ,'flexible-pricing':5, 'co-operative-decision-making':6, 'no-strategy':7}

        if role_id==0:
            return tourist_strategy[pred_strategy]
        else:
            return agent_strategy[pred_strategy]
            

    def get_meteor_score(self, candidates, current_sentence):

        meteor_score_list = []
        
        for i in candidates:
            reference = []
            candidate = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            predicted = word_tokenize(candidate) 
            ref = word_tokenize(current_sentence)
            reference.append(ref)
            meteor_score = round(meteor(reference, predicted),2)  
            meteor_score_list.append(meteor_score)         
        return meteor_score_list 


    def get_utt_t_score(self, candidates, turn_num, dial_inputs):

        utt_t_list = []
        
        for i in candidates:
            candidate = self.tokenizer.decode(i.tolist()[0][2:]).split('\n')[0].split('\t')[0]
            if(turn_num>=2):
                topic = self.tokenizer.decode(dial_inputs[turn_num-1].tolist()[0]).split('\n')[0].split('\t')[3].strip()
            else:
                topic = ''
            turn = []
            turn.append(candidate)
            turn.append(topic)
            turn=model.encode(turn)
            score = cosine_similarity([turn[0]], turn[1:])[0][0]
            utt_t_list.append(score)
        
        return utt_t_list
    
    def strategy_pred(self,current_sentence,pred_strategy,role_id):
  
        strategy_reward = []

        for i in range(len(current_sentence)):
            idx = self.label2idx(pred_strategy[i],role_id)

            if role_id ==0:
                strategy_re = 0.54
                strategy_reward.append(strategy_re)
            if role_id == 1:
                strategy_re=0.6
                strategy_reward.append(strategy_re)
        return strategy_reward


    def top_p_candidates(self,logits, prob=0.92, filter_value=-float('Inf')):
    
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cum_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cum_sum > prob
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter_(1, index=sorted_indices, src=sorted_indices_to_remove.clone())
        logits[indices_to_remove] = filter_value
        
        return logits

    def get_logits_output(self,logits):
        next_token_logits = logits[:, -1, :].contiguous() / 0.8
        next_token_logits = self.top_p_candidates(next_token_logits, 0.92)
        next_token_log_probs = F.log_softmax(next_token_logits, -1)
        probs = F.softmax(next_token_logits, dim=-1)
                
        next_tokens = torch.multinomial(probs, num_samples=1)
        next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
        next_tokens = next_tokens.squeeze(1)
        return next_tokens.cpu()

    def get_predicted_strategy(self,sentence,role_id):
    
        tourist_strategy = {'na':-1,'problem-solving':0, 'strategic-proposal':1 , 'firm-pricing':2, 'definitive-decision-making':3,'collaborative-proposal':4 ,'flexible-pricing':5, 'co-operative-decision-making':6, 'no-strategy':7}
        agent_strategy = {'na':-1,'problem-solving':0, 'strategic-proposal':1 , 'firm-pricing':2, 'definitive-decision-making':3,'collaborative-proposal':4 ,'flexible-pricing':5, 'co-operative-decision-making':6, 'no-strategy':7}

        
        sent = sentence.split(' ')
        sent = [tok.strip() for tok in sent if len(tok)>0]
        get_strategy = False
        strategy = ''
        
        for i in range(len(sent)):
            if sent[i] == 'strategy:':
                get_strategy = True
            if get_strategy == True:
                strategy = sent[i]
                get_strategy = False
            if i<len(sent)-1:
                if sent[i]=='strategy' and sent[i+1] == 'of':
                    get_strategy = True
                if get_strategy==True and sent[i]!='of':
                    strategy = sent[i]
                    get_strategy = False
        if role_id == 0:
            if strategy in agent_strategy.keys():
                return strategy
            else:
                return 'NA'
        elif role_id==1:
            if strategy in tourist_strategy.keys():
                return strategy
            else:
                return 'NA'


    def slice_input(self,dial_turn_input):
        slice_len = 0
        for i in range(dial_turn_input.shape[1]):
            if dial_turn_input[0][i] == 197:
                return slice_len
            else:
                slice_len +=1
        return slice_len

    def validate_model(self, dataloader):
        print("Validation Step")
        f = open('exp1_output_full.txt','a')
        with torch.no_grad():
            if not self.train_single_model:
                self.model_A.eval()
                self.model_B.eval()
            else:
                if self.single_model_to_train == 'agent':
                    self.model_A.eval()
                else:
                    self.model_B.eval()

            with torch.no_grad():
                
                progress_bar = tqdm
                pbar = progress_bar(dataloader)
               
                total_ppl = []
                total_loss = []
                total_r_len = []
                total_meteor = []
                total_strategy_reward = []
                total_prize_gap_reward = []
                context = []
                for batch in pbar:
                    role_ids, _, strategies, intents, stages, personas = batch[0]
                    r_id = []
                    conv_list=[]
                    strategy_list = []
                    intent_list = []
                    stage_list = []
                    persona_list = []
                    ind=0
                    br_flag = False
                    last_item = len(batch[0][1])-1
                    last_rid = len(role_ids)-1
                    print("Last item, rid index",last_item,last_rid)
                    if sum([len(item) for item in batch[0][1]]) > 1024:
                        
                        trim_indx = 0
                        
                        for item in batch[0][1]:
                            conv = item
                            print("Added: ",ind)
                            
                            
                            trim_indx = trim_indx + len(conv)
                            if trim_indx<1024 - len(batch[0][1][last_item]):
                                conv_list.append(conv)
                                ind=ind+1
                            else:
                                br_flag = True
                                break

                        for i in range(ind):
                            r_id.append(role_ids[i])
                        if br_flag==True:
                            print("Last_item_added")
                            conv_list.append(batch[0][1][last_item])
                            r_id.append(role_ids[last_rid])
                        
                        batch[0] = (r_id,conv_list,strategy_list,intent_list,stage_list,persona_list)


                    role_ids, dialog_tokens, dialog_strategy, dialog_intent, dialog_stage, dialog_persona = batch[0]



                    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]
                    dial_strategies = [torch.LongTensor(item).unsqueeze(0) for item in dialog_strategy]
                    dial_intents = [torch.LongTensor(item).unsqueeze(0) for item in dialog_intent]
                    dial_stages = [torch.LongTensor(item).unsqueeze(0) for item in dialog_stage]
                    dial_personas = [torch.LongTensor(item).unsqueeze(0) for item in dialog_persona]
                    past = None
                    past_ = None
                    all_logits = []
                    target = []
                    conv_sentences = []
                    final_conv = len(dial_inputs)-1
                    i=0
                    final_intent = ''
                    for turn_num, dial_turn_inputs in enumerate(dial_inputs):
                        
                        current_sentence = self.tokenizer.decode(dial_turn_inputs.tolist()[0][2:]).split('\t')[0]
                        next_sentence = tokenizer.decode(dial_turn_inputs.tolist()[0][2:]).split('\t')[1]
                        
                        
                        if not self.train_single_model:
                            if role_ids[turn_num] == 0:
                                index = self.slice_input(dial_turn_inputs)
                                
                                new_input = dial_turn_inputs[:,0:index]
                                input_act = convert_sentences_to_strings([dial_turn_inputs], self.tokenizer)[0]
                                new_input_str = convert_sentences_to_strings([new_input], self.tokenizer)[0]

                                f.write('Actual Input')
                                f.write('\n')
                                f.write(input_act)
                                f.write('\n')
                                f.write('new_input')
                                f.write(new_input_str)

                                dial_turn_inputs = dial_turn_inputs.to(self.device)
                                outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                
                                index = self.slice_input(dial_turn_inputs)
                                new_input = dial_turn_inputs[:,0:index]
                                generated_sequence, generated_log_probs  = generate_n_candidates(self.model_A,
                                                                                          new_input,
                                                                                          self.top_p,
                                                                                          eos_token_id=self.turn_ending[0],
                                                                                          pad_token_id=self.turn_ending[1],
                                                                                          num_candidates=self.num_candidates,
                                                                                          max_gen_length=200,
                                                                                          temperature=self.temperature,
                                                                                          past=past_,
                                                                                          device=self.device)
                                pred_sentence = convert_sentences_to_strings(generated_sequence, self.tokenizer)
                                p_sent = pred_sentence[0]
                                f.write('\n')
                                f.write('predicted_sentence')
                                f.write(p_sent)

                                pred_strategy = []
                                buyer_p = []
                                seller_min_pr = []
                                seller_init_price = []
                                
                                past = outputs[1]
                                all_logits.append(outputs[0])
                            else:
                                
                                dial_turn_inputs = dial_turn_inputs.to(self.device)
                                outputs = self.model_B(dial_turn_inputs, past_key_values=past, return_dict=False)
                                index = self.slice_input(dial_turn_inputs)
                                new_input = dial_turn_inputs[:,0:index]

                                input_act = convert_sentences_to_strings([dial_turn_inputs], self.tokenizer)[0]
                                new_input_str = convert_sentences_to_strings([new_input], self.tokenizer)[0]
                                f.write('\n')
                                f.write('Actual Input')
                                f.write('\n')
                                f.write(input_act)
                                f.write('\n')
                                f.write('new_input')
                                f.write(new_input_str)

                                generated_sequence, generated_log_probs  = generate_n_candidates(self.model_A,
                                                                                          new_input,
                                                                                          self.top_p,
                                                                                          eos_token_id=self.turn_ending[0],
                                                                                          pad_token_id=self.turn_ending[1],
                                                                                          num_candidates=self.num_candidates,
                                                                                          max_gen_length=200,
                                                                                          temperature=self.temperature,
                                                                                          past=past_,
                                                                                          device=self.device)
                                pred_sentence = convert_sentences_to_strings(generated_sequence, self.tokenizer)
                                pred_strategy = []
                                for sent in pred_sentence:

                                    pred_strategy.append(self.get_predicted_strategy(sent,role_ids[turn_num]))
                                past = outputs[1]
                                all_logits.append(outputs[0])
                        else:
                            if self.single_model_to_train == 'agent':
                                if role_ids[turn_num] == 0:
                                    dial_turn_inputs = dial_turn_inputs.to(self.device)
                                    index = self.slice_input(dial_turn_inputs)
                                    new_input = dial_turn_inputs[:,0:index]
                                    input_act = convert_sentences_to_strings([dial_turn_inputs], self.tokenizer)[0]
                                    outputs = self.model_A(dial_turn_inputs, past_key_values=past, return_dict=False)
                                    past = outputs[1]
                                    all_logits.append(outputs[0])
                                    target.append(dial_turn_inputs)
                                    generated_sequence, generated_log_probs  = generate_n_candidates(self.model_A,
                                                                                          new_input,
                                                                                          self.top_p,
                                                                                          eos_token_id=self.turn_ending[0],
                                                                                          pad_token_id=self.turn_ending[1],
                                                                                          num_candidates=self.num_candidates,
                                                                                          max_gen_length=200,
                                                                                          temperature=self.temperature,
                                                                                          past=past_,
                                                                                          device=self.device)
                                    output = self.model_A(expand_inputs_for_N_candidates(dial_turn_inputs,
                                                                                         self.num_candidates),
                                                                                         past_,
                                                                                         return_dict=False)
                                    past_ = output[1]
                                    current_sentence = self.tokenizer.decode(dial_turn_inputs.tolist()[0][2:]).split('\t')[0]
                        i = i+1

                    all_logits = torch.cat(all_logits, dim=1)
                    all_logits = all_logits[:, :-1].contiguous()

                    if not self.train_single_model:
                        target = torch.cat(dial_inputs, dim=1)[:, 1:].contiguous()
                    else:
                        target = torch.cat(target, dim=1)[:, 1:].contiguous()
                    
                    target_mask = torch.ones_like(target).float()

                    loss = self.criterion(all_logits.to('cuda'), target.to('cuda'), target_mask.to('cuda'), label_smoothing=-1, reduce='sentence')
                    total_loss.extend(loss.tolist())

                    ppl = torch.exp(loss)
                    total_ppl.extend(ppl.tolist())
                    


                    average_lengths = self.get_candidate_lengths(generated_sequence)
                    total_r_len.append(np.mean(average_lengths))

                    meteor_scores = self.get_meteor_score(generated_sequence, current_sentence)
                    total_meteor.append(np.mean(meteor_scores))


                print('\n')
                print(f"Validation Perplexity: {np.mean(total_ppl)}")

                print(f"Overall Average candidate length: {np.mean(total_r_len)}")
                print(f"Overall Meteor score: {np.mean(total_meteor)}")
                context.append(current_sentence)

        return np.mean(total_ppl), np.mean(total_loss), np.mean(average_lengths)
    

    def make_stats_dir(self):
        
        self.statsfolder = os.path.join(os.getcwd(), self.savefolder, self.modelname, 'stats')
        if not os.path.exists(self.statsfolder):
            os.makedirs(self.statsfolder)


    def make_model_save_dir(self):
        self.savefolder = os.path.join(os.getcwd(), 'Planner_sample')
        if not os.path.exists(self.savefolder):
            print("Model save folder doesn't exist.")
            os.makedirs(self.savefolder)
            print(f"Created folder {self.savefolder} to save the models.")


    def save_models(self, num_iter):
        
        modeldir = os.path.join(self.savefolder, self.modelname)
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
            print('Created Directory for saving models!')

        filenameA = modeldir + '/' + self.modelname + '_A' + str(num_iter) + ".pth"
        filenameB = modeldir + '/' + self.modelname + '_B' + str(num_iter) + ".pth"
        torch.save(self.model_A.state_dict(), filenameA)
        torch.save(self.model_B.state_dict(), filenameB)

    def modified_train_one_iter(self, batch):
        dial_inputs, role_ids, scores_dict = collect_samples(batch,
                                                             model_A=self.model_A_ref,
                                                             model_B=self.model_B,
                                                             top_p=self.top_p,
                                                             eos_token_id=self.turn_ending[0],
                                                             pad_token_id=self.turn_ending[1],
                                                             average_sent_loss=self.average_sent_loss,
                                                             max_gen_length=self.max_candidate_length,
                                                             buffer_memory=self.buffer_memory,
                                                             use_persona=self.use_persona,
                                                             use_strategy=self.use_strategy,
                                                             use_stage_intent_consistency=self.use_stage_intent_consistency,
                                                             use_diversity=self.use_diversity,
                                                             use_contextual_consistency=self.use_contextual_consistency,
                                                             use_fluency=self.use_fluency,
                                                             device=self.device,
                                                             num_candidates=self.num_candidates,
                                                             human_reward=self.human_reward,
                                                             tokenizer=self.tokenizer,
                                                             criterion=self.criterion,
                                                             temperature=self.temperature,
                                                             use_recent_past=self.use_recent_past,
                                                             recompute_log_prob=self.recompute_log_prob,
                                                             nlp=self.nlp,
                                                             train_single_model=self.train_single_model,
                                                             model_to_train=self.single_model_to_train,
                                                             gamma1=self.gamma1,
                                                             gamma2=self.gamma2,
                                                             gamma3=self.gamma3,
                                                             gamma4=self.gamma4,
                                                             gamma5=self.gamma5,
                                                             gamma6=self.gamma6)

        log_dict = ppo_step(model_A=self.model_A,
                            model_B=self.model_B,
                            buffer_memory=self.buffer_memory,
                            train_single_model=self.train_single_model,
                            dial_inputs= dial_inputs,
                            model_to_train=self.single_model_to_train,
                            device=self.device,
                            ppo_epsilon=self.epsilon,
                            num_candidates=self.num_candidates,
                            use_recent_past=self.use_recent_past,
                            average_sent_loss=self.average_sent_loss,
                            criterion=self.criterion,
                            optimizer=self.optimizer,
                            role_ids=role_ids)

        self.buffer_memory.clear_memory()

        return log_dict, scores_dict 
 
    def train(self):

        update_count = 0
        progress_bar = tqdm

        val_ppl = []
        val_loss = []

        rewards = []
        kl = []
        clip_frac = []

        persona_scores = []
        strategy_scores = []
        stage_intent_consistency_scores = []
        diversity_scores = []
        contextual_consistency_scores = []
        fluency_scores = []
        

        best_ppl = None
        
        length = None
        
        iters = None
        
        progress_bar = tqdm
        pbar = progress_bar(self.train_dataloader)
        

        print('Training Model for',self.n_epochs,' epochs.')
        for i in tqdm(range(self.n_epochs)):
            if not self.train_single_model:
                self.model_A.train()
                self.model_B.train()
            else:
                if self.single_model_to_train == 'agent':
                    self.model_A.train()
            
            update_list = []

            for batch in pbar:

                role_ids, _, strategies, intents, stages, personas = batch[0]
                r_id = []
                conv_list = []
                strategy_list = []
                intent_list = []
                stage_list = []
                persona_list = []
                ind=0
                br_flag = False
                last_item = len(batch[0][1])-1
                last_rid = len(role_ids)-1
                if sum([len(item) for item in batch[0][1]]) > 1024-300:
                    
                    trim_indx = 0
                    
                    for item in batch[0][1]:
                        conv = item
                        
                        
                        
                        trim_indx = trim_indx + len(conv)
                        if trim_indx<1024 - len(batch[0][1][last_item])-300:
                            conv_list.append(conv)
                            ind=ind+1
                        else:
                            br_flag = True
                            break

                    for j in range(ind):
                        r_id.append(role_ids[j])
                    
                    for k in range(ind):
                        strategy_list.append(strategies[k])
                    for k in range(ind):
                        intent_list.append(intents[k])
                    for k in range(ind):
                        stage_list.append(stages[k])
                    for k in range(ind):
                        persona_list.append(personas[k])


                    
                    batch[0] = (r_id,conv_list,strategy_list,intent_list,stage_list,persona_list)


                print(f"ITERATION: {update_count}")
                print("Epoch: ",i)

                batch = batch[0]
                

                
                log_dict, scores_dict  = self.modified_train_one_iter(batch)

                clip_frac.append(log_dict['clip_frac'])
                kl.append(log_dict['approx_kl'])
                rewards.append(log_dict['reward'])

                persona_scores.extend(scores_dict['persona'])
                strategy_scores.extend(scores_dict['strategy'])
                stage_intent_consistency_scores.extend(scores_dict['stage_intent_consistency'])
                diversity_scores.extend(scores_dict['diversity'])
                contextual_consistency_scores.extend(scores_dict['contextual_consistency'])
                fluency_scores.extend(scores_dict['fluency'])

                

                np.save(self.statsfolder + '/' + 'persona_scores.npy', np.array(persona_scores))
                np.save(self.statsfolder + '/' + 'strategy_scores.npy', np.array(strategy_scores))
                np.save(self.statsfolder + '/' + 'stage_intent_consistency_scores.npy', np.array(stage_intent_consistency_scores))
                np.save(self.statsfolder + '/' + 'diversity_scores.npy', np.array(diversity_scores))
                np.save(self.statsfolder + '/' + 'contextual_consistency_scores.npy', np.array(contextual_consistency_scores))
                np.save(self.statsfolder + '/' + 'fluency_scores.npy', np.array(fluency_scores))

            

                update_count += 1

                print('update count is:', update_count)

                if  update_count % self.evaluate_every == 0:
                    print('Validating Model')
                    

                    update_list.append(update_count)

                    
                    ppl, loss, average_length = self.validate_model(self.val_dataloader)
                    

                    
                    
                    
                    if best_ppl is None:

                        best_ppl = ppl
                        iters = update_count
                        
                        length = average_length
                        
                        if update_count >= 20:
                          self.save_models(iters)
                          print(f'Saving Model at {iters}')
                        
                    else:
                        if ppl < best_ppl:
                            best_ppl = ppl
                            iters = update_count
                            
                            length = average_length
                            
                        if update_count >= 20:
                          self.save_models(iters)
                          print(f'Saving Model at {iters}')
                
                    print('\n')
                    print(f'Best Perplexity Found so far {best_ppl} for iteration: {iters}')
                    print('\n')
                    
                    val_ppl.append(ppl)
                    val_loss.append(loss)
                    
                                
                    np.save(self.statsfolder + '/' + 'val_PPL_iter'  + '.npy', np.array(val_ppl))
                    np.save(self.statsfolder + '/' + 'train_rewards' + '.npy', np.array(rewards))
                    np.save(self.statsfolder + '/' + 'train_kl' + '.npy', np.array(kl))
                    np.save(self.statsfolder + '/' + 'train_clip_frac' + '.npy', np.array(clip_frac))
                    np.save(self.statsfolder + '/' + 'best_ppl_iteration_value' + '.npy', np.array(iters))
                    


                    if not self.train_single_model:
                        self.model_A.train()
                        self.model_B.train()
                    else:
                        if self.single_model_to_train == 'agent':
                            self.model_A.train()


                
        return best_ppl, iters

class NegotiationDataset(Dataset):
    def __init__(self, data, next_strategy, next_utter, next_intent, next_stage, next_persona, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        self.next_utter = next_utter
        self.next_strategy = next_strategy
        self.next_intent = next_intent
        self.next_stage = next_stage
        self.next_persona = next_persona
        # tokenizer weird behavior
        self.turn_ending = [628, 198]


        # tokenizer.encode("\n\n\n")
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, index):
        dial_tokens = []
        dial_strategy = []
        dial_stage = []
        dial_intent = []
        dial_persona = []
        test_persona = []
        
        for i in range(len(self.data[index])):
            print('i = ',i)
            sep1 = "\t"
            
            if self.data[index][i][0]=='A':
                sep1 = "next agent response \t"
               
            elif self.data[index][i][0]=='B':
                sep1 = "next toursist response \t"

            else:
                print("### Not Getting ###")
                print(self.data[index][i])

            sep2 = '||'
            sep3 = '||'
            sep4 = '||'
            sep4 = '||'
            sep5 = '||' 
            
            # print('self.data[index][i]',self.data[index][i])
            # print('self.next_utter[index][i]',self.next_utter[index][i])
            # print('self.next_strategy[index][i]',self.next_strategy[index][i])
            # print('self.next_intent[index][i]',self.next_intent[index][i])
            # print('self.next_stage[index][i]',self.next_stage[index][i])
            # print('self.next_persona[index][i]',self.next_persona[index][i])


            dial_tokens.append(self.tokenizer.encode(self.data[index][i]) + self.tokenizer.encode(sep1) + self.tokenizer.encode(self.next_utter[index][i]) + self.tokenizer.encode(sep2) + self.tokenizer.encode(self.next_strategy[index][i])  + self.tokenizer.encode(sep3) + self.tokenizer.encode(self.next_intent[index][i]) + self.tokenizer.encode(sep4) + self.tokenizer.encode(self.next_stage[index][i])  + self.tokenizer.encode(sep5) + self.tokenizer.encode(self.next_persona[index][i]) + self.turn_ending)
            dial_strategy.append(self.tokenizer.encode(self.data[index][i].split('||')[1]))
            dial_intent.append(self.tokenizer.encode(self.data[index][i].split('||')[2]))
            dial_stage.append(self.tokenizer.encode(self.data[index][i].split('||')[3]))
            dial_persona.append(self.tokenizer.encode(self.data[index][i].split('||')[4]))
            test_persona.append(self.data[index][i].split('||')[1])
            
        
        
        role_ids = [0 if item[0] == 32 else 1 for item in dial_tokens]
        # print('self.data[index]',self.data[index])
        # print('dial_strategy',dial_strategy)
        # print('dial_intent',dial_intent)
        # print('dial_stage',dial_stage)
        # print('dial_persona',dial_persona)
        # print('test_persona',test_persona)
        # print('role_ids',role_ids)
        return role_ids, dial_tokens, dial_strategy, dial_intent, dial_stage, dial_persona
        
    def collate(self, unpacked_data):
        return unpacked_data


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("cuda--in--use")
    else:
        print("no cuda available!!!!")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device_type: ',device)
    trainer = Trainer(modelname='model_v1',
                      train_csvfile="../planner_final/data/sample/deal_train_sample.csv",
                      val_csvfile="../planner_final/data/sample/deal_valid_sample.csv",
                      device=torch.device("cuda"),
                      n_epochs=10,
                      batch_size=8,
                      mini_batch=10,
                      train_single_model=False,
                      single_model_to_train = 'agent',
                      num_candidates=3,
                      recompute_log_prob=True,
                      average_sent_loss=True,
                      max_candidate_length=300,
                      human_reward=10,
                      top_p=0.9,
                      temperature=0.8,
                      use_recent_past=True,
                      warmup_steps=10,
                      print_every=1,
                      evaluate_every=20,
                      learning_rate=2e-5,
                      epsilon=0.2,
                      loadModel=False,
                      loadFilenameA="gpt_ppo_model",
                      loadFilenameB="gpt_ppo_model",
                      pad_token_id=2,
                      seedvalue=10, # 10 should be the seed value since pre trained on the same seed. 
                      use_persona=True,
                      use_strategy=True,
                      use_stage_intent_consistency=True,
                      use_diversity=True,
                      use_contextual_consistency=True,
                      use_fluency=True,
                      gamma1=0.3,
                      gamma2=0.2,
                      gamma3=0.3,
                      gamma4=0.1,
                      gamma5=0.1,
                      gamma6=0.1
                      )

    trainer.train()
    print('Training Done')
