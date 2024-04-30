import numpy as np
import pandas as pd
import torch.nn as nn
from nltk.translate.meteor_score import meteor_score
import nltk
import re
#nltk.download('wordnet')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools
import operator
import pdb
import spacy
# from deepcopy import copy
from sklearn.metrics.pairwise import cosine_similarity
#from bert_score import score
from evaluate import load
from sentence_transformers import SentenceTransformer
#from persona_nli import *
import warnings
from get_strategy_new import get_prediction
from get_stage_new import get_stage_prediction

from stage_intent_model_new import evaluate_stage_intent_model
from persona_nli import get_nli

import numpy as np
from collections import Counter
import math
warnings.filterwarnings("ignore")
from transformers import GPT2LMHeadModel, GPT2Tokenizer


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

model = SentenceTransformer('bert-base-nli-mean-tokens')

GPT_model = GPT2LMHeadModel.from_pretrained('microsoft/DialoGPT-medium').to('cuda')
GPT_tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-medium')

GPT_tokenizer.pad_token = GPT_tokenizer.eos_token

GPT_model.eval()

MAX_LEN = 64

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def load_stage_intent_model():
    args = torch.load('../planner/dialogpt/stage-intent-model/training_args.bin')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    stage_intent_model = AutoModelWithLMHead.from_pretrained("../planner/dialogpt/stage-intent-model")
    stage_intent_model.to('cuda')
    return args,tokenizer,stage_intent_model


def convert_sentences_to_strings(sentences:list, tokenizer):
    str_sentences = []
    for i in sentences:
        str_sentences.append(tokenizer.decode(i.tolist()[0][2:-2])) # Excludes the zero shot tokens: {A:, B:} and the End of turn tokens: [628, 198]
    return str_sentences

def normalize(text, nlp):
    sent = ''
    doc = nlp(text)
    for token in doc:
        if not token.is_punct:
            sent += token.lemma_
            sent += ' '
    return sent

def label2idx(pred_strategy,role_id):
  tourist_strategy = {'na':-1,'problem-solving':0, 'strategic-proposal':1 , 'firm-pricing':2, 'definitive-decision-making':3,'collaborative-proposal':4 ,'flexible-pricing':5, 'co-operative-decision-making':6, 'no-strategy':7}
  agent_strategy = {'na':-1,'problem-solving':0, 'strategic-proposal':1 , 'firm-pricing':2, 'definitive-decision-making':3,'collaborative-proposal':4 ,'flexible-pricing':5, 'co-operative-decision-making':6, 'no-strategy':7}

  if role_id==0:
    return tourist_strategy[pred_strategy.lower()]
  else:
    return agent_strategy[pred_strategy.lower()]

def label2idx_stage(pred_stage,role_id):
    tourist_stage = {'relational_positioning':0, 'problem_identification':1, 'offer_generation':2, 'decision_making':3}
    agent_stage = {'relational_positioning':0, 'problem_identification':1, 'offer_generation':2, 'decision_making':3}
    if role_id==0:
        return tourist_stage[pred_stage.lower()]
    else:
        return agent_stage[pred_stage.lower()]
    
def strategy_pred(candidates,actual_strategy_label,role_id, delta=0.2):
    strategy_reward = []
    sen_len = []
    print("Inside strategy_pred_module")
    print('candidates',len(candidates))
    print('actual_strategy_label',actual_strategy_label)
    print('role_id',role_id)

    for i in range(len(candidates)):
        indx = candidates[i].find('next')
        candidates[i] = candidates[i][indx:].replace('\n',' ')
    print('============')
    print('candidates[0]',candidates[0])
    print('candidates[1]',candidates[1])
    print('candidates[2]',candidates[2])
    
    actual_label_prob = 0.0
    other_label_prob = 0.0

    for i in range(len(candidates)):
        if actual_strategy_label != 'NA':
            idx = label2idx(actual_strategy_label.lower(),role_id)
            strategy_probs = get_prediction(candidates[i],idx,role_id)
            actual_label_prob = strategy_probs[:, idx][0]
            other_label_prob = (strategy_probs.sum(-1) - actual_label_prob)[0]

        
        strategy_re = actual_label_prob - delta * other_label_prob
        strategy_reward.append(strategy_re)    
    
  
    
    return strategy_reward

def calculate_persona_alignment(candidates, num_turn, dial_inputs, tokenizer,actual_persona_profile):
    persona_alignment_scores= []
    persona_list = []
    score = 0.0
    for i in candidates:
        candidate_sentence = i.replace('\n',' ')
        
        if len(actual_persona_profile) != 0:
            prob, _ = get_nli(candidate_sentence, actual_persona_profile)
            entail_prob = [score[1].item() + score[2].item() for score in prob]
            score = sum(entail_prob)
            
        persona_alignment_scores.append(score)

    return persona_alignment_scores

def jacc_sim(context_sentence_list, generated_sentences, tokenizer, nlp):

    temp1 = context_sentence_list[0].split('||')
    index = temp1[4].find('next')
    str1 = temp1[0] + ' ' + temp1[4][index:]
    str1 = normalize(str1, nlp)
    str1 = set(str1.split())
    jacc_dis = []
    for i in generated_sentences:
        str2 = i
        str2 = normalize(str2, nlp)
        str2 = set(str2.split())
        sim_score = 1-(float(len(str1 & str2)) / len(str1 | str2))
        jacc_dis.append(sim_score)
    return jacc_dis

def calculate_contextual_consistency(candidates, current_sentence, tokenizer, num_turn, dial_inputs):
    contextual_consistency_scores = []

    bertscore = load("bertscore")

    for i in candidates:
        candidate_sentence = i.replace('\n',' ')
        if(num_turn>=2):
            prev_sentence = tokenizer.decode(dial_inputs[num_turn-1].tolist()[0][2:]).replace('\n',' ')
            prev_sentence1 = tokenizer.decode(dial_inputs[num_turn-2].tolist()[0][2:]).replace('\n',' ')
        else:
            prev_sentence = ''

        # with (i, r)
        turn = []
        turn.append(candidate_sentence)
        turn.append(current_sentence)
        
        results = bertscore.compute(predictions=[turn[0]], references=[turn[1]], lang="en")
        score1 = results['f1'][0]
        
        # with (i-1, r)
        turn = []
        turn.append(candidate_sentence)
        turn.append(prev_sentence)
    

        results = bertscore.compute(predictions=[turn[0]], references=[turn[1]], lang="en")
        score2 = results['f1'][0]
        
        sim_score = 0.5*(score1+score2)

        contextual_consistency_scores.append(sim_score)
    
    return contextual_consistency_scores

# Perplexity
def perplexity(candidates):

    perplexity_scores = []

    for i in candidates:

        candidate_sentence = i

        BATCH_SIZE = 1

        tokenized_input = GPT_tokenizer.batch_encode_plus(candidate_sentence, max_length=MAX_LEN, pad_to_max_length=True, truncation=True)
        
        input_ids = tokenized_input['input_ids'] 
        attention_masks = tokenized_input['attention_mask']

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        data = TensorDataset(input_ids, attention_masks)

        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size = BATCH_SIZE)

        all_loss = []

        with torch.no_grad():

            for batch in dataloader:
                b_input = batch[0].to('cuda')
                b_attn = batch[1].to('cuda')

                outputs = GPT_model(b_input, attention_mask=b_attn, labels=b_input)
        
                loss, logits = outputs[:2]
                all_loss.append(loss.item())

        ppl_score = math.exp(np.mean(all_loss))

        perplexity_scores.append(ppl_score)
    return perplexity_scores

def calculate_stage_intent_consistency(generated_sentences, current_sentence, next_sentence, actual_intent_label, actual_stage_label, num_turn, dial_inputs, tokenizer,eta = 0.1):
    utt_stage_int_scores = []
    scores = []

    label2stage = {0:'relational_positioning', 1:'problem_identification', 2:'offer_generation', 3:'decision_making'}

    for candidate in generated_sentences:
        list1 = []
        list1.append(current_sentence.split('||')[0] +' [INTENT] '+actual_intent_label) #tourist_response concat intent
        list2=[]
        list2.append(next_sentence.split('||')[0] +' [STAGE] '+actual_stage_label) #agent_gold_response concat stage

        stage_re = get_stage_prediction(candidate)

        
        pred_stage = label2stage[stage_re]
        list3=[]
        list3.append(candidate +' [STAGE] '+pred_stage) #agent_predicted_response concat predicted stage
        

        eval_data1 = []
        eval_data1.append(list1)
        eval_data1.append(list2)
        eval_data2= []
        eval_data2.append(list1)
        eval_data2.append(list3)

        eval_df1 = pd.DataFrame({'response':eval_data1[1],'context':eval_data1[0],'context/0':'.','context/1':'.','context/2':'.','context/3':'.'})
        eval_df2 = pd.DataFrame({'response':eval_data2[1],'context':eval_data2[0],'context/0':'.','context/1':'.','context/2':'.','context/3':'.'})

        eval_res1 = evaluate_stage_intent_model(eval_df1)
        eval_res2 = evaluate_stage_intent_model(eval_df2)

        nll1 = eval_res1['loss']
        nll2 = eval_res2['loss']
        
        score = nll1 - eta * nll2
        score = -(torch.tanh(torch.tensor(score)))
        scores.append(score)
        utt_stage_int_scores.append(score.item())
    
    return utt_stage_int_scores




def calculate_rewards(model_A, 
                      current_sentence,
                      next_sentence,
                      current_sentence_intent,
                      next_sentence_stage,
                      actual_strategy_label,
                      actual_intent_label,
                      actual_stage_label,
                      actual_persona_profile,
                      num_turn,
                      dial_inputs,
                      length,
                      generated_sentences,
                      source_list,
                      conv_sentences,
                      tokenizer,
                      criterion,
                      pred_strategy,
                      role_id,
                      use_persona,
                      use_strategy,
                      use_stage_intent_consistency,
                      use_diversity,
                      use_contextual_consistency,
                      use_fluency,
                      nlp,
                      device,
                      gamma1,
                      gamma2,
                      gamma3,
                      gamma4,
                      gamma5,
                      gamma6,
                      agent=False):

    
    scores = {}

    scores['persona'] = []
    scores['strategy'] = []
    scores['stage_intent_consistency'] = []
    scores['diversity'] = []
    scores['contextual_consistency'] = []
    scores['fluency'] = []

    persona_alignment_scores = 0
    strategy_logit = 0
    stage_intent_scores = 0
    diversity = 0
    contextual_consistency = 0
    fluency = 0
    
    if len(generated_sentences) >= 1:
        
        rewards = np.zeros(len(generated_sentences))

            
        if use_strategy:
            strategy_scores = strategy_pred(generated_sentences,actual_strategy_label,role_id)
            strategy_scores = np.array(strategy_logit)
            
            rewards += gamma2*(strategy_scores)
        else: 
            strategy_logit = np.zeros(len(generated_sentences))
        
        
        if use_persona:
            persona_alignment_scores = calculate_persona_alignment(generated_sentences, num_turn, dial_inputs, tokenizer, actual_persona_profile)
            rewards += gamma1*np.array(persona_alignment_scores)
        
        else:
            persona_alignment_scores = np.zeros(len(generated_sentences))

        if use_stage_intent_consistency:
            usi_cons_scores = calculate_stage_intent_consistency(generated_sentences, current_sentence, next_sentence, actual_intent_label, actual_stage_label, num_turn, dial_inputs, tokenizer)
            rewards+= gamma3*np.array(usi_cons_scores)
        else: 
            usi_cons_scores = np.zeros(len(generated_sentences))


        if use_diversity: 
            non_rep = jacc_sim(source_list, generated_sentences, tokenizer, nlp)
            
            non_rep = np.array(non_rep)

            diversity = 1 - non_rep

            rewards -= gamma4*(diversity)
        else: 
            diversity = np.zeros(len(generated_sentences))

        if use_contextual_consistency:
            contextual_consistency_scores = calculate_contextual_consistency(generated_sentences, current_sentence, tokenizer, num_turn, dial_inputs)
            rewards += gamma5*np.array(contextual_consistency_scores)
        
        else:
            contextual_consistency_scores = np.zeros(len(generated_sentences))

        if use_fluency:
            perplexity_scores = perplexity(generated_sentences)
            fluency_scores = [1/f for f in perplexity_scores]
            rewards += gamma6*np.array(fluency_scores)
        else:
            fluency_scores = np.zeros(len(generated_sentences))


    else:
        rewards = 0
    
    scores['persona'].extend([persona_alignment_scores])
    scores['strategy'].extend([strategy_logit])
    scores['stage_intent_consistency'].extend([usi_cons_scores])
    scores['diversity'].extend([diversity])
    scores['contextual_consistency'].extend([contextual_consistency_scores])
    scores['fluency'].extend([fluency_scores])
    
    return list(rewards), scores

def append(generated_list, context_sentence, tokenizer):
    
    if len(generated_list) == 2:
        generated_list.pop(0)
        cntx = tokenizer.decode(context_sentence.tolist()[0][2:]).split('\n')[0]
        generated_list.append(cntx)
    else:
        cntx = tokenizer.decode(context_sentence.tolist()[0][2:]).split('\n')[0]
        generated_list.append(cntx)
    
    return generated_list

def expand_inputs_for_N_candidates(inputs, num_candidates):
    # inputs = inputs[None, ...]
    return inputs.repeat((num_candidates, 1))

def modify_generated_sequence(generated_sequences, generated_log_probs):
    
    final_generated_sequences = []
    final_generated_log_probs = []
    
    for i in range(generated_sequences.shape[0]):
        
        batch_tokens = []
        batch_log_probs = []
        
        for j in range(len(generated_sequences[i])):
            if generated_sequences[i][j] != 628 and generated_sequences[i][j] != -1:
                batch_tokens.append(generated_sequences[i][j])
                batch_log_probs.append(generated_log_probs[i][j])
            elif generated_sequences[i][j] == 628:
                batch_tokens.append(generated_sequences[i][j])
                batch_log_probs.append(generated_log_probs[i][j])
                batch_tokens.append(198)
                break
            else:
                break
        final_generated_sequences.append(torch.tensor(batch_tokens).unsqueeze(0))
        ### BE CAREFUL WHEN USING THIS, SINCE IT DOESN NOT AVERAGES THE LOG PROBS INSTEAD IT JUST TAKES THE SUM.
        final_generated_log_probs.append(torch.tensor(batch_log_probs).sum().item())
    
    return final_generated_sequences, final_generated_log_probs

def top_p_candidates(logits, prob=0.92, filter_value=-float('Inf')):
    
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cum_sum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cum_sum > prob
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter_(1, index=sorted_indices, src=sorted_indices_to_remove.clone())
    logits[indices_to_remove] = filter_value
    
    return logits

def generate_n_candidates(model,
                          inputs,
                          top_p,
                          temperature,
                          num_candidates,
                          max_gen_length,
                          past,
                          device,
                          eos_token_id=628,
                          pad_token_id=198):

    curr_len = inputs.shape[1]
    if inputs.shape[1]>200:

        max_gen_length=300
    else:
        max_gen_length=210
    inputs = expand_inputs_for_N_candidates(inputs, num_candidates)
    inputs_ = inputs
    generated_sequences = torch.ones((inputs.shape[0], max_gen_length), dtype=torch.long) * -1
    generated_sequences[:, 0:inputs.shape[1]] = inputs.cpu()
    
    generated_token_log_prob = torch.zeros((inputs.shape[0], max_gen_length), dtype=torch.float)
    
    unfinished_sequences = inputs.new(inputs.shape[0]).fill_(1) #.cpu()
    
    i = 0
    
    while True:
        if past:
            if past[0][0].shape[-2] > 1024:
                if not torch.all(generated_sequences==-1):
                    final_generated_sequence, final_generated_log_probs = modify_generated_sequence(generated_sequences, generated_token_log_prob)
                    return final_generated_sequence, final_generated_log_probs, past
                else:
                    return None, None
        
        outputs = model(inputs, past)
        logits, past = outputs[0], outputs[1]
        
        next_token_logits = logits[:, -1, :].contiguous() / temperature
        
        if top_p and top_p > 0.0:
            # This returns score after performing softmax function.
            next_token_logits = top_p_candidates(next_token_logits, top_p)
            next_token_log_probs = F.log_softmax(next_token_logits, -1)
            probs = F.softmax(next_token_logits, dim=-1)
            
            next_tokens = torch.multinomial(probs, num_samples=1)
            next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
            next_tokens = next_tokens.squeeze(1)
            
            if eos_token_id is not None:
                assert pad_token_id is not None # "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # NOTE: SAVE LOG PROBS AS WELL
            generated_sequences[:, curr_len] = next_tokens.cpu()
            inputs = next_tokens.unsqueeze(1).to(device)
            #inputs_ = torch.cat((inputs_, next_tokens[:, None]), dim=-1)
            
            curr_len = curr_len + 1
            
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break
            if curr_len >= max_gen_length:
                break
    
    final_generated_sequences, final_generated_log_probs =  modify_generated_sequence(generated_sequences, generated_token_log_prob)
    
    return final_generated_sequences, final_generated_log_probs

def compute_log_probs(target_token_ids, logits, mask, average_sent_loss=False):
    logits = logits[:, :-1, :].contiguous() # (batch, sequence_length, vocab_size)
    
    target_token_ids = target_token_ids[:, 1:].contiguous() # (batch, sequence_length)
    

    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, -1, target_token_ids.unsqueeze(-1)).squeeze(-1)
    mask = mask[:, 1:].contiguous()
    
    if average_sent_loss:
        log_probs = (log_probs * mask).sum(-1) / mask.sum(-1)
    else:
        log_probs = (log_probs * mask).sum(-1)
    return {'log_probs': log_probs}

def get_logits_output(logits):
    next_token_logits = logits[:, -1, :].contiguous() / 0.8
    next_token_logits = top_p_candidates(next_token_logits, 0.92)
    next_token_log_probs = F.log_softmax(next_token_logits, -1)
    probs = F.softmax(next_token_logits, dim=-1)
            
    next_tokens = torch.multinomial(probs, num_samples=1)
    next_token_log_probs = next_token_log_probs.gather(-1, next_tokens)
    next_tokens = next_tokens.squeeze(1)
    return next_tokens.cpu()

def get_predicted_strategy(sent3,role_id):
    
    tourist_strategy = {'problem-solving':0, 'strategic-proposal':1, 'firm-pricing':2, 'definitive-decision-making':3,'collaborative-proposal':4 ,'flexible-pricing':5, 'co-operative-decision-making':6, 'no-strategy':7}
    agent_strategy = {'problem-solving':0, 'strategic-proposal':1, 'firm-pricing':2, 'definitive-decision-making':3,'collaborative-proposal':4 ,'flexible-pricing':5, 'co-operative-decision-making':6, 'no-strategy':7}

    indx = sent3.find('next')
    sent3 = sent3[indx:]
    sent = sent3.lower()
    sent = sent.split(' ')
    sent = [k.strip() for k in sent]
    sent.reverse()
    Flag = False
    if role_id==0:
        key_li = tourist_strategy.keys()
    else:
        key_li = agent_strategy.keys()
    strategy = 'NA'
    key_li = [ele.lower() for ele in key_li]
    for ele in sent:
      for k in key_li:
        if k in ele:
          Flag = True
          el_len = len(ele.split('_'))
          k_len = len(k.split('_'))
          if k_len == el_len:
            Flag = True
            strategy = k
            break
          
        
      if Flag == True:
        break
    
    return strategy
   

def ppo_step(model_A,
             model_B,
             buffer_memory,
             device,
             ppo_epsilon,
             num_candidates,
             criterion,
             optimizer,
             dial_inputs,
             role_ids,
             scheduler=None,
             train_single_model=False,
             model_to_train=None,
             average_sent_loss=False,
             use_recent_past=False):

    optimizer.zero_grad()
    
    log_dict = {}
    
    new_log_prob = []
    old_log_prob = []
    
    rewardlist = []
    
    ratios = []
    
    policy_loss = []
    advantages  = []

    if use_recent_past:
        print('USING RECENT PAST')
    else:
        print('NOT USING RECENT PAST')

    if use_recent_past:
        
        batches = buffer_memory.get_batch(shuffle=False)
        
        past = None
        
        i = 0
        
        for idx, batch in enumerate(batches):
            
            action = torch.tensor(batch['action'], device=device).unsqueeze(0)
            if batch['human_response']:
                
                if idx == 0:
                    logits, past = model_A(action, past, return_dict=False)
                
                if idx > 0 and idx % (num_candidates + 1) == 0:
                    try:
                        past = out
                    except:
                        pass
                    
                    if i < len(dial_inputs):
                        history = dial_inputs[i]
                    else:
                        continue
                    
                    _, past = model_A(history.to(device), past_key_values=None, return_dict=False)
                    logits, out = model_A(action, past_key_values=past, return_dict=False)
                    
                    i += 2
            else:
                history_indices = idx // (num_candidates + 1)  # {A:(1,2,3,4,5),B, C:(7,8,9,10,11), D, E: (13,14,15,16,17)}
                
                if history_indices == 0:
                    logits, _ = model_A(action, past_key_values=None, return_dict=False)
                else:
                    logits, _ = model_A(action, past_key_values=past, return_dict=False)
            
            new_log_probs = compute_log_probs(target_token_ids=action,
                                              logits=logits,
                                              mask=torch.ones_like(action).to(device),
                                              average_sent_loss=average_sent_loss)['log_probs']

            old_log_probs = torch.tensor(batch['log_prob'], device=device).unsqueeze(0)
            old_log_prob.append(old_log_probs)

            rewards = torch.tensor(batch['reward'], device=device).unsqueeze(0)
            rewardlist.append(batch['reward'])
            advantages.append(rewards)

            new_log_prob.append(new_log_probs)

        if new_log_prob:
            new_log_prob = torch.cat(new_log_prob, dim=-1)
            old_log_prob = torch.cat(old_log_prob, dim=-1)
        
            advantages = torch.cat(advantages, dim=-1)
        
            ratio = (new_log_prob - old_log_prob).exp()
        
            policyloss1 = - advantages * ratio
            policyloss2 = - advantages * ratio.clamp(1 - ppo_epsilon, 1 + ppo_epsilon)
        
            policyloss = torch.min(policyloss1, policyloss2).mean()
        
            policyloss.backward()

            with torch.no_grad():
                log_dict['policy_loss'] = policyloss.item()
                print('Policy Loss: ', log_dict['policy_loss'])
                
                # (r-1) - logr, where r = p(x)/q(x); p(x) = new distribution and q(x) is old distribution
                log_dict['approx_kl'] = torch.mean(((new_log_prob - old_log_prob).exp() - 1)\
                                                - (new_log_prob - old_log_prob)).item()
                #log_dict['approx_kl'] = 0.5 * np.mean(np.power((np.array(new_log_prob) - np.array(old_log_prob)), 2))
                print('approx KL div: ', log_dict['approx_kl'])
                
                log_dict['clip_frac'] = torch.mean((torch.abs(ratio-1) > ppo_epsilon).float()).item()
                print('clip frac: ', log_dict['clip_frac'])
                
                log_dict['reward'] = np.mean(rewardlist)
                print('rewards: ', log_dict['reward'])
        else:
            log_dict['policy_loss'] = 0
            print('Policy Loss: ', log_dict['policy_loss'])
                
            # (r-1) - logr, where r = p(x)/q(x); p(x) = new distribution and q(x) is old distribution
            log_dict['approx_kl'] = 0
            
            #log_dict['approx_kl'] = 0.5 * np.mean(np.power((np.array(new_log_prob) - np.array(old_log_prob)), 2))
            print('approx KL div: ', log_dict['approx_kl']) 

            log_dict['clip_frac'] = 0
            print('clip frac: ', log_dict['clip_frac'])
                
            log_dict['reward'] = 0
            print('rewards: ', log_dict['reward'])
        

    if not train_single_model:
        nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)
        nn.utils.clip_grad_norm_(model_B.parameters(), 1.0)
    else:
        if model_to_train =='agent':
            nn.utils.clip_grad_norm_(model_A.parameters(), 1.0)

    optimizer.step()
    #scheduler.step()

    return log_dict
def slice_input(dial_turn_input):
    slice_len = 0
    #print('shape of turn 1: ',dial_turn_input.shape[1])
    for i in range(dial_turn_input.shape[1]):
        if dial_turn_input[0][i] == 197:
            #print("tensor: ",dial_turn_input[0][i])
           # print("*****")
            return slice_len
        else:
            ##print("tensor: ",dial_turn_input[0][i])
            #print("else")
            slice_len +=1
    return slice_len


@torch.no_grad()
def collect_samples(batch,
                    model_A,
                    model_B,
                    top_p,
                    eos_token_id,
                    pad_token_id,
                    max_gen_length,
                    num_candidates,
                    human_reward,
                    use_persona,
                    use_strategy,
                    use_stage_intent_consistency,
                    use_diversity,
                    use_contextual_consistency,
                    use_fluency,
                    buffer_memory,
                    device,
                    tokenizer,
                    criterion,
                    temperature,
                    use_recent_past,
                    average_sent_loss,
                    nlp,
                    gamma1,
                    gamma2,
                    gamma3,
                    gamma4,
                    gamma5,
                    gamma6,
                    train_single_model=True,
                    model_to_train=None,
                    recompute_log_prob=True,
                    fp16=False):

    scores_dict = {}

    scores_dict['persona'] = []
    scores_dict['strategy'] = []
    scores_dict['stage_intent_consistency'] = []
    scores_dict['diversity'] = []
    scores_dict['contextual_consistency'] = []
    scores_dict['fluency'] = []
    

    
    print("In training Step")

    role_ids, dialog_tokens, dialog_strategy, dialog_intent, dialog_stage, dialog_persona = batch

   
    
    dial_inputs = [torch.LongTensor(item).unsqueeze(0) for item in dialog_tokens]
    dial_strategies = [torch.LongTensor(item).unsqueeze(0) for item in dialog_strategy]
    dial_intents = [torch.LongTensor(item).unsqueeze(0) for item in dialog_intent]
    dial_stages = [torch.LongTensor(item).unsqueeze(0) for item in dialog_stage]
    dial_personas = [torch.LongTensor(item).unsqueeze(0) for item in dialog_persona]
    

    pastA = None
    pastB = None
    past = None
    past_ = None
    past_A = None
    past_B = None
    context = None
    cntxt = None

    agent_generated_list, user_generated_list = [], []
    length = np.zeros(num_candidates)
    length = length.tolist()

    conv_sentences = []
    final_conv = len(dial_inputs)-1
    
    
    i=0
    t=0
    final_strategy = ''
    for num_turn, dialog_turn_inputs in enumerate(dial_inputs):
        print("#### loop i",t)
        t+=1
        assert not np.any(np.isnan(dialog_turn_inputs).cpu().numpy()), 'Inputs Dialog contains Nan value.'
        
        dialog_turn_inputs = dialog_turn_inputs.to(device)
        
        current_sentence = tokenizer.decode(dialog_turn_inputs.tolist()[0][2:]).split('\t')[0]
        next_sentence = tokenizer.decode(dialog_turn_inputs.tolist()[0][2:]).split('\t')[1]

        if not train_single_model:
            if role_ids[num_turn] == 0:
                #pastA = None
                outputs = model_A(dialog_turn_inputs, past, return_dict=False)
                logits = outputs[0]
                index = slice_input(dialog_turn_inputs)
                new_input = dialog_turn_inputs[:,0:index]
                mask = torch.ones_like(dialog_turn_inputs).to(device)
                generated_sequence, generated_log_probs = generate_n_candidates(model_A,
                                                                new_input, top_p,
                                                                eos_token_id=eos_token_id,
                                                                pad_token_id=pad_token_id,
                                                                num_candidates=num_candidates,
                                                                max_gen_length=max_gen_length,
                                                                temperature=temperature,
                                                                past=past_,
                                                                device=device)
                pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)

                pred_strategy = []
                log_probs = compute_log_probs(target_token_ids=dialog_turn_inputs,
                                              logits=logits,
                                              mask=mask,
                                              average_sent_loss=average_sent_loss)
                
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                            context=context,
                                            action=dialog_turn_inputs.tolist()[0],
                                            action_log_probs=log_probs['log_probs'].item(),
                                            reward=human_reward,
                                            agent=True,
                                            human_response=True)
                if not use_recent_past:
                    '''In this case, first we generate sentence using the entire past. And then we update the past with
                    the current utterance.'''
                
                    generated_sequence, generated_log_probs  = generate_n_candidates(model_A,
                                                                                     new_input,
                                                                                     top_p,
                                                                                     eos_token_id=eos_token_id,
                                                                                     pad_token_id=pad_token_id,
                                                                                     num_candidates=num_candidates,
                                                                                     max_gen_length=max_gen_length,
                                                                                     temperature=temperature,
                                                                                     past=past_,
                                                                                     device=device)
                    pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)[0]
                    print('pred_sentence2',pred_sentence)
                    exit()
                    pred_strategy = get_predicted_strategy(pred_sentence,role_ids[num_turn])
                    pred_strategy = []
                    for sent in pred_sentence:

                        pred_strategy.append(get_predicted_strategy(sent,role_ids[num_turn]))
                        
                    
                    output = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs,num_candidates),
                                     past_,
                                     return_dict=False)

                    past_ = output[1]
                else:
                    generated_sequence, generated_log_probs = generate_n_candidates(model_A,
                                                                                    torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device), top_p,
                                                                                    eos_token_id=eos_token_id,
                                                                                    pad_token_id=pad_token_id,
                                                                                    num_candidates=num_candidates,
                                                                                    max_gen_length=max_gen_length,
                                                                                    temperature=temperature,
                                                                                    past=past_,
                                                                                    device=device)
            else:
                outputs = model_B(dialog_turn_inputs, past, return_dict=False)
                logits = outputs[0]
                sent_logit = get_logits_output(logits)
                index = slice_input(dialog_turn_inputs)
                new_input = dialog_turn_inputs[:,0:index]
                mask = torch.ones_like(dialog_turn_inputs).to(device)
                generated_sequence, generated_log_probs = generate_n_candidates(model_B,
                                                                new_input, top_p,
                                                                eos_token_id=eos_token_id,
                                                                pad_token_id=pad_token_id,
                                                                num_candidates=num_candidates,
                                                                max_gen_length=max_gen_length,
                                                                temperature=temperature,
                                                                past=past_,
                                                                device=device)
                pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)
                pred_strategy = []
                log_probs = compute_log_probs(target_token_ids=dialog_turn_inputs,
                                              logits=logits,
                                              mask=mask,
                                              average_sent_loss=average_sent_loss)
                
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                            context=context,
                                            action=dialog_turn_inputs.tolist()[0],
                                            action_log_probs=log_probs['log_probs'].item(),
                                            reward=human_reward,
                                            agent=True,
                                            human_response=True)
                
                if not use_recent_past:
                    '''In this case, first we generate sentence using the entire past. And then we update the past with
                    the current utterance.'''
                    #past_ = None
                    generated_sequence, generated_log_probs  = generate_n_candidates(model_B,
                                                                                     new_input,
                                                                                     top_p,
                                                                                     eos_token_id=eos_token_id,
                                                                                     pad_token_id=pad_token_id,
                                                                                     num_candidates=num_candidates,
                                                                                     max_gen_length=max_gen_length,
                                                                                     temperature=temperature,
                                                                                     past=past_,
                                                                                     device=device)
                    pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)
                    
                    pred_strategy = []
                    for sent in pred_sentence:

                        pred_strategy.append(get_predicted_strategy(sent,role_ids[num_turn]))
                        
                    output = model_B(expand_inputs_for_N_candidates(dialog_turn_inputs,num_candidates),
                                     past_,
                                     return_dict=False)

                    past_ = output[1]
                else:
                    generated_sequence, generated_log_probs = generate_n_candidates(model_B,
                                                                                        torch.tensor(tokenizer.encode("A:")).unsqueeze(0).to(device), top_p,
                                                                                        eos_token_id=eos_token_id,
                                                                                        pad_token_id=pad_token_id,
                                                                                        num_candidates=num_candidates,
                                                                                        max_gen_length=max_gen_length,
                                                                                        temperature=temperature,
                                                                                        past=past_,
                                                                                        device=device)

        elif model_to_train == 'agent':
            
            
            if role_ids[num_turn] == 0:
                
                '''if use_recent_past:
                    if cntxt is not None:
                        past = prepare_inputs(cntxt, model_A)
                    else:
                        past = None'''
                
                
                outputs = model_A(dialog_turn_inputs, pastA, return_dict=False)
                logits = outputs[0]
                mask = torch.ones_like(dialog_turn_inputs).to(device)
                index = slice_input(dialog_turn_inputs)
                new_input = dialog_turn_inputs[:,0:index]
                input_act = convert_sentences_to_strings([dialog_turn_inputs], tokenizer)[0]

                
                log_probs = compute_log_probs(target_token_ids=dialog_turn_inputs,
                                              logits=logits,
                                              mask=mask,
                                              average_sent_loss=average_sent_loss)
                
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                            context=context,
                                            action=dialog_turn_inputs.tolist()[0],
                                            action_log_probs=log_probs['log_probs'].item(),
                                            reward=human_reward,
                                            agent=True,
                                            human_response=True)
                
                if not use_recent_past:
                    '''In this case, first we generate sentence using the entire past. And then we update the past with
                    the current utterance.'''
                
                    generated_sequence, generated_log_probs  = generate_n_candidates(model_A,
                                                                                     new_input,
                                                                                     top_p,
                                                                                     eos_token_id=eos_token_id,
                                                                                     pad_token_id=pad_token_id,
                                                                                     num_candidates=num_candidates,
                                                                                     max_gen_length=max_gen_length,
                                                                                     temperature=temperature,
                                                                                     past=past_,
                                                                                     device=device)
                    pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)[0]
                    output = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs,num_candidates),
                                     past_,
                                     return_dict=False)
                    '''Here first we calculate the past based on the context sentence and then we generate candidates.'''
                    '''if cntxt is not None:
                        past_ = prepare_inputs(expand_inputs_for_N_candidates(cntxt, num_candidates), model_A)
                    else:
                        past_ = None'''
                    
                else:

                    generated_sequence, generated_log_probs = generate_n_candidates(model_A,
                                                                                    new_input, top_p,
                                                                                    eos_token_id=eos_token_id,
                                                                                    pad_token_id=pad_token_id,
                                                                                    num_candidates=num_candidates,
                                                                                    max_gen_length=max_gen_length,
                                                                                    temperature=temperature,
                                                                                    past=past_,
                                                                                    device=device)
                    pred_sentence = convert_sentences_to_strings(generated_sequence, tokenizer)[0]
                    
        gen_sent = convert_sentences_to_strings(generated_sequence, tokenizer)
        
        agent_generated_list = append(agent_generated_list, dialog_turn_inputs, tokenizer)
        
        if num_turn+1<len(dial_inputs):
            next_sentence = tokenizer.decode(dial_inputs[num_turn+1].tolist()[0][2:])
        else:
            next_sentence = ''

        
        actual_intent_label = tokenizer.decode(dial_intents[num_turn][0].tolist())
        actual_persona_profile = list(eval(tokenizer.decode(dial_personas[num_turn][0].tolist())))
        
        
        if num_turn+1<len(dial_inputs):
            print('Inside actual_strategy_label if')
            actual_strategy_label = tokenizer.decode(dial_strategies[num_turn+1][0].tolist())
            actual_stage_label = tokenizer.decode(dial_stages[num_turn+1][0].tolist())
        else: 
            print('Inside actual_strategy_label else')
            actual_strategy_label = 'NA'
            actual_persona_profile = []


        
        current_sentence_strategy = current_sentence.split('||')[1]
        current_sentence_intent = current_sentence.split('||')[2]
        
        if next_sentence != '':
            next_sentence_stage_temp = next_sentence.split('||')[3]
            next_sentence_stage = next_sentence_stage_temp.split('next tourist response')[0]
        else: 
            next_sentence_stage = ''

        

        reward, scores = calculate_rewards(current_sentence=current_sentence,
                                            next_sentence=next_sentence,
                                            current_sentence_intent=current_sentence_intent,
                                            next_sentence_stage=next_sentence_stage,
                                            actual_strategy_label=actual_strategy_label,
                                            actual_intent_label=actual_intent_label,
                                            actual_stage_label=actual_stage_label,
                                            actual_persona_profile=actual_persona_profile,
                                            num_turn=num_turn,
                                            dial_inputs=dial_inputs,
                                            generated_sentences= pred_sentence,
                                            length=length,
                                            source_list=agent_generated_list,
                                            tokenizer=tokenizer,
                                            criterion=criterion,
                                            agent=True,
                                            role_id=role_ids[num_turn],
                                            conv_sentences=conv_sentences,
                                            use_persona=use_persona,
                                            use_strategy=use_strategy,
                                            use_stage_intent_consistency=use_stage_intent_consistency,
                                            use_diversity=use_diversity,
                                            use_contextual_consistency=use_contextual_consistency,
                                            use_fluency=use_fluency,
                                            pred_strategy=pred_strategy,
                                            nlp=nlp,
                                            device=device,
                                            gamma1=gamma1,
                                            gamma2=gamma2,
                                            gamma3=gamma3,
                                            gamma4=gamma4,
                                            gamma5=gamma5,
                                            gamma6=gamma6,
                                            model_A=model_A)

        print('persona:',scores['persona'])
        print('strategy:',scores['strategy'])
        print('stage_intent_consistency',scores['stage_intent_consistency'])
        print('diversity',scores['diversity'])
        print('contextual_consistency',scores['contextual_consistency'])
        print('fluency scores',scores['fluency'])

        scores_dict['persona'].extend(scores['persona'])
        scores_dict['strategy'].extend(scores['strategy'])
        scores_dict['stage_intent_consistency'].extend(scores['stage_intent_consistency'])
        scores_dict['diversity'].extend(scores['diversity'])
        scores_dict['contextual_consistency'].extend(scores['contextual_consistency'])
        scores_dict['fluency'].extend(scores['fluency'])

        
        conv_sentences.append(current_sentence)
        if recompute_log_prob:

            for j in range(len(generated_sequence)):
                
                # NOTE: STILL USING THE PAST FROM PREVIOUS UTTERANCE, SINCE WE DO NOT NEED PAST FROM
                #       CONTAINING CURRENT UTTERANCE for GENERATED CANDIDATES
                if role_ids[num_turn] == 0:
                    output = model_A(generated_sequence[j].to(device), past_key_values=past, return_dict=False)
                    logits = output[0]
                    
                    log_probs = compute_log_probs(target_token_ids=generated_sequence[j].to(device),
                                                logits=logits,
                                                mask=torch.ones_like(generated_sequence[j]).to(device),
                                                average_sent_loss=average_sent_loss)['log_probs'].item()
                    
                    buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                                context=context,
                                                action= generated_sequence[j].tolist()[0],
                                                action_log_probs=log_probs,
                                                reward=reward[j],
                                                agent=True,
                                                human_response=False)
                else:
                    output = model_B(generated_sequence[j].to(device), past_key_values=past, return_dict=False)
                    logits = output[0]
                    
                    log_probs = compute_log_probs(target_token_ids=generated_sequence[j].to(device),
                                                logits=logits,
                                                mask=torch.ones_like(generated_sequence[j]).to(device),
                                                average_sent_loss=average_sent_loss)['log_probs'].item()
                    
                    buffer_memory.update_buffer(state=dialog_turn_inputs.tolist()[0],
                                                context=context,
                                                action= generated_sequence[j].tolist()[0],
                                                action_log_probs=log_probs,
                                                reward=reward[j],
                                                agent=True,
                                                human_response=False)

        else:
            for k in range(len(generated_sequence)):
                buffer_memory.update_buffer(state=dialog_turn_inputs.tolis()[0],
                                            action=generated_sequence[k].tolist()[0],
                                            action_log_probs=generated_log_probs[k],
                                            agent=True,
                                            human_response=False)
        if role_ids[num_turn] == 0:
            past = outputs[1]
            #past = None
            outputs = model_A(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
            past_ = outputs[1]
            #past_ = None
            #Q
        else:
            past = outputs[1]
            #past = None
            #past = outputs[1]
            outputs = model_B(expand_inputs_for_N_candidates(dialog_turn_inputs, num_candidates), past_, return_dict=False)
            past_ = outputs[1]
            #past_ = None

    
        
        context = dialog_turn_inputs.tolist()[0]
        cntxt = dialog_turn_inputs
        
        i=i+1

    return dial_inputs, role_ids, scores_dict, #candidate_dict

def get_past(batches, model, device):
    
    states = torch.cat(batches, dim=-1).to(device)
    outputs = model(states, past_key_values=None, return_dict=False)
    
    return outputs[1]

def prepare_inputs_for_model(batches, model, num_candidates, device):
    
    states = get_history_utterances(batches, num_candidates)
    states = torch.cat(states, dim=1, device=device)
    outputs = model(states, past_key_values=None, return_dict=False)
    
    return outputs[1]

def get_history_utterances(batches, num_candidates):
    states = []
    for i in range(0, len(batches), num_candidates+1):
        states.append(i)
    return states

def get_recursive_past(dial_inputs, role_ids, model_A, model_B, device):
    '''
    Uses both models alternatively to calculate pasts.
    Used in case of training only the agent.
    '''
    past = None
    for num_turn, utter in enumerate(dial_inputs):
        if role_ids[num_turn] == 0:
            _, past = model_A(utter.to(device), past_key_values=past, return_dict=False)
        else:
            _, past = model_B(utter.to(device), past_key_values=past, return_dict=False)
    return past
