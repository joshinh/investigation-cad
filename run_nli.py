# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (c) 2020 Sawan Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""
#


from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer)

from transformers import (BertConfig,
                          BertForSequenceClassification,
                          BertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup

from example_to_feature import convert_examples_to_features as convert_examples_to_features

from utils import ExpProcessor
from utils import exp_compute_metrics as compute_metrics

#from lm_utils import NLIDataset, CoQADataset

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (RobertaConfig,)), ())

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logits(batch, model_output, exp_model):
    if exp_model in ["instance", "append", "instance_append", "all_explanation", "Explanation_1", "hyp_only"]:
        return model_output
    model_output = model_output.view(batch[0].size(0), batch[0].size(1), model_output.size(1))
    e,c,n = 0,1,2
    v1,v2 = 0,1
    if exp_model in ["independent", "instance_independent"]:
        evidence_e = [model_output[:, e, v1]]
        evidence_c = [model_output[:, c, v1]]
        evidence_n = [model_output[:, n, v1]]
    elif exp_model in ["aggregate", "instance_aggregate"]:
        evidence_e = [model_output[:, e, v1], model_output[:, c, v2]]
        evidence_c = [model_output[:, e, v2], model_output[:, c, v1]]
        evidence_n = [model_output[:, n, v1]]
    logits_e, logits_c, logits_n = [torch.cat([evd.unsqueeze(-1) for evd in item], 1)
            for item in [evidence_e, evidence_c, evidence_n]]
    logits_e, logits_c, logits_n = [torch.logsumexp(item, 1, keepdim=True) for item in [logits_e, logits_c, logits_n]]
    logits = torch.cat([logits_e, logits_c, logits_n], 1)
    return logits

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total) 
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    erm_loss, logging_erm_loss, con_loss, logging_con_loss = 0.0, 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_result = None
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
            # Remove dummy examples
            if args.contrast:
                input_ids = batch[0].view(-1, batch[0].size(-1))
                attention_mask = batch[1].view(-1, batch[0].size(-1))
                labels = batch[3].view(-1)
                dummy_mask = (torch.sum(input_ids, dim=-1) > 0).long().to(args.device)
                input_ids = input_ids[torch.nonzero(dummy_mask, as_tuple=True)[0],:].long()
                attention_mask = attention_mask[torch.nonzero(dummy_mask, as_tuple=True)[0],:].long()
                labels = labels[torch.nonzero(dummy_mask, as_tuple=True)[0]].long()
                indices = batch[4].view(-1)[torch.nonzero(dummy_mask, as_tuple=True)[0]].long()
            else:
                input_ids = batch[0].view(-1, batch[0].size(-1))
                attention_mask = batch[1].view(-1, batch[0].size(-1))
                token_type_ids = batch[2].view(-1, batch[0].size(-1))
                
            
            inputs = {'input_ids': input_ids,
                      'attention_mask': attention_mask}
            if args.model_type == 'roberta':
                inputs['token_type_ids'] = None
            elif args.model_type == 'bert':
                inputs['token_type_ids'] = token_type_ids
            inputs['token_type_ids'] = None
            outputs = model(**inputs)
            model_output = outputs[0]
            
            if args.contrast:
                # Weighted Cross Entropy
                
#                 loss_fct = nn.NLLLoss()
#                 logits = get_logits(batch, model_output, args.exp_model)
#                 log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
#                 target = batch[3].view(-1)
                
#                 pred = torch.argmax(logits, dim=-1)
#                 correct = (target == pred).long()
#                 consistent = torch.tensor([torch.sum(x)/5.0 for x in torch.split(correct, int(correct.shape[0]/batch[3].shape[0]))])
#                 consistent[consistent == 0.0] = 1.0
#                 inverse_weights = torch.repeat_interleave(1.0 / consistent, 5).to(args.device)
#                 per_example_weight = torch.ones_like(pred).float().to(args.device)
#                 per_example_weight = torch.where(correct == 0, inverse_weights, per_example_weight)
                
#                 loss = loss_fct(log_probs * per_example_weight.unsqueeze(-1), target)

                # Pairwise consistency (also includes mse with original, which is always zero)
                loss_fct = nn.CrossEntropyLoss()
                logits = get_logits(batch, model_output, args.exp_model)
                ce_loss = loss_fct(logits, labels)
                
                alpha = 10.0
                consistency_regularizer = nn.MSELoss()
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                log_probs = torch.gather(log_probs, dim=-1, index=labels.view(-1,1)).squeeze(-1)
                
                # Only use regularization when pairs are available
                index_mask = (indices < 8330).long().to(args.device)
                log_probs = log_probs[torch.nonzero(index_mask, as_tuple=True)[0]]
                orig_log_probs = torch.repeat_interleave(log_probs[::5], 5).to(args.device)
                
                if torch.sum(index_mask) > 0:
                    consistency_loss = consistency_regularizer(log_probs, orig_log_probs)
                else:
                    consistency_loss = torch.tensor(0.0)
                
                loss = ce_loss + (alpha * consistency_loss) 
                
                erm_loss += ce_loss.item()
                con_loss += consistency_loss.item()
                
            else:
                loss_fct = nn.CrossEntropyLoss()
                logits = get_logits(batch, model_output, args.exp_model)
                loss = loss_fct(logits, batch[3])

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    
                    if args.contrast:
                        tb_writer.add_scalar('erm_loss', (erm_loss - logging_erm_loss)/args.logging_steps, global_step)
                        tb_writer.add_scalar('con_loss', (con_loss - logging_con_loss)/args.logging_steps, global_step)
                        logging_erm_loss = erm_loss
                        logging_con_loss = con_loss
                        
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
                
        #Save and evaluate after every epoch
        if args.save_every_epoch:
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            results = evaluate(args, model, tokenizer, prefix="", analyze_attentions=False, eval_on_train=False, epoch=str(epoch))
            if best_result is None:
                best_result = results
                best_result["ckpt"] = epoch
            if best_result["acc"] < results["acc"]:
                best_result["ckpt"] = epoch
                best_result["acc"] = results["acc"]
                
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
        
    if best_result is not None:
        src_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(best_result["ckpt"]))
        tgt_dir = os.path.join(args.output_dir, 'best-checkpoint')
        shutil.copytree(src_dir, tgt_dir)
        logger.info("Saving best model checkpoint from %s to %s", src_dir, tgt_dir)

    return global_step, tr_loss / global_step

def is_two_class(data_dir):
    if 'hans' in data_dir:
        return True
    else:
        return False


def evaluate(args, model, tokenizer, prefix="", analyze_attentions=False, eval_on_train=False, epoch=''):
    processor = ExpProcessor()
    eval_output_dir = args.output_dir

    results = {}
    eval_dataset, indices = load_and_cache_examples(args, tokenizer, evaluate=True)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    attentions = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0].view(-1, batch[0].size(-1)),
                      'attention_mask': batch[1].view(-1, batch[0].size(-1))}
            inputs['token_type_ids'] = None
            outputs = model(**inputs)
            model_output = outputs[0]
            loss_fct = nn.CrossEntropyLoss()
            logits = get_logits(batch, model_output, args.exp_model)
            tmp_eval_loss = loss_fct(logits, batch[3])
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        inputs['labels'] = batch[3]
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds_ = preds
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids, is_two_class(args.eval_file))
    results.update(result)

    if not eval_on_train:
        eval_dataset = os.path.basename(os.path.dirname(args.eval_file)).split('_')[1]
        eval_file_base = os.path.splitext(os.path.basename(args.eval_file))[0]
        to_drop_list = args.to_drop.split(',') if evaluate else []
        to_drop_str = '_drop'+''.join(to_drop_list) if args.to_drop else ''
        epoch_str = '_' + epoch if epoch != '' else ''
        prediction_file = os.path.join(args.output_dir, 'predictions_{}_{}_{}_{}{}{}.npz'.format(
            eval_dataset,
            eval_file_base,
            str(args.max_seq_length),
            str(args.data_format),
            to_drop_str,
            epoch_str))
        print ("Writing predictions to ", prediction_file)
        np.savez_compressed(prediction_file, dist=preds_, preds=preds, indices=indices, labels=out_label_ids) 

    output_eval_file = os.path.join(eval_output_dir, "eval_results_{}_{}{}.txt".format(
        eval_dataset,
        eval_file_base,
        epoch_str))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file

    filename = args.train_file if not evaluate else args.eval_file
    data_dir, filename_base = os.path.split(filename)
    two_class = is_two_class(data_dir)
    
    processor = ExpProcessor(two_class)

    if args.data_format == "aggregate": data_storage_format = "independent"
    elif args.data_format == "instance_aggregate": data_storage_format = "instance_independent"
    else: data_storage_format = args.data_format

    to_drop_list = args.to_drop.split(',') if evaluate else []
    
    cached_features_file = os.path.join(data_dir, 'cached_seq{}_{}_{}_{}_{}'.format(
        str(args.max_seq_length),
        filename_base,
        data_storage_format,
        'drop'+args.to_drop if args.to_drop and evaluate else '',
        '_negs' if args.sample_negs and not evaluate else ''
        ))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        examples, features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        fn = processor.get_dev_examples if evaluate else processor.get_train_examples
        examples = fn(filename, data_format=data_storage_format, to_drop=to_drop_list)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                pad_on_left=False,
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save((examples, features), cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    features = features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    
    indices = [example.guid for example in examples]
    
    # Group data for contrast sets
    # Assumption - size of each set is 5
    if args.contrast and not evaluate:
        total_size = all_input_ids.shape[0]
        num_contrast_sets = int(total_size/5)
        
        # Add dummy data points if total size is not mod 5
        if len(all_labels) % 5 != 0:
            extra = 5 - len(all_labels) % 5
            all_input_ids = torch.cat((all_input_ids, torch.zeros(extra, all_input_ids.shape[-1])), dim=0)
            all_attention_mask = torch.cat((all_attention_mask, torch.zeros(extra, all_attention_mask[-1].shape[-1])), dim=0)
            all_token_type_ids = torch.cat((all_token_type_ids, torch.zeros(extra, all_token_type_ids[-1].shape[-1])), dim=0)
            all_labels = torch.cat((all_labels, torch.zeros(extra)), dim=0)
            indices.extend([len(indices) + x for x in range(extra)])
        
        all_input_ids = all_input_ids.view(-1,5, all_input_ids.shape[-1])
        all_attention_mask = all_attention_mask.view(-1,5,all_attention_mask.shape[-1])
        all_token_type_ids = all_token_type_ids.view(-1,5,all_token_type_ids.shape[-1])
        all_labels = all_labels.view(-1,5)
        indices = torch.tensor(indices, dtype=torch.long).view(-1,5)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, indices)
    else:
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    return dataset, indices


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="The input train file. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_file", default=None, type=str, required=True,
                        help="The input eval file. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--exp_model", default="instance", type=str)
    parser.add_argument("--data_format", default="instance", type=str)
    parser.add_argument("--to_drop", default="", type=str) #comma-sep list

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--prompt_type", default="none", type=str,
                        help="Prompt given before explanation")
    parser.add_argument("--use_annotations", action='store_true',
                        help="Whether to use annotations instead of generated explanations")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--sample_negs', action='store_true',
                        help='sample negative conjectures')

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--contrast', action='store_true',
                       help='use contrast sets based training')
    parser.add_argument('--save_every_epoch', action='store_true',
                       help='store model weights after every epoch')
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    processor = ExpProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.exp_model in ["independent", "instance_independent"]: args.model_num_outputs = 1
    elif args.exp_model in ["aggregate", "instance_aggregate"]: args.model_num_outputs = 2
    else: args.model_num_outputs = 3

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=args.model_num_outputs,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case,
                                            cache_dir=args.cache_dir if args.cache_dir else None)

    if args.do_train:
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        model.to(args.device)
        train_dataset, indices = load_and_cache_examples(args, tokenizer, evaluate=False)
        
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.output_hidden_states = True
            model.output_attentions = True
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step, analyze_attentions=True)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    print (results)
    return results


if __name__ == "__main__":
    main()
