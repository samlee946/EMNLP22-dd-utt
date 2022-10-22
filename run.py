import json
import logging
import numpy as np
import os
import random
import sys
import time
import torch
from datetime import datetime
from os.path import join
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW

import conll
import helper
import util
from metrics import CorefEvaluator, MentionDetectionEvaluator
from model import CorefModel
from tensorize import CorefDataProcessor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


# logger.addHandler(logging.FileHandler(join('/users/sxl180006/research/codi2021/datadir_dd', 'log_batch_non_refer.txt'), 'a'))

class Runner:
    def __init__(self, config_name, gpu_id=0, best_model_suffix=None, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed
        self.best_model_suffix = best_model_suffix

        # Set up config
        self.config = util.initialize_config(config_name)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)

        # Set up seed
        if seed:
            util.set_seed(self.seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        self.data = CorefDataProcessor(self.config)

        # Set up seed
        if seed:
            util.set_seed(self.seed)

    def set_test_file(self, test_file):
        self.config['test_dir'] = test_file

    def initialize_model(self, saved_suffix=None):
        model = CorefModel(self.config, self.device)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def initialize_model_from_state_dict(self, state_dict):
        model = CorefModel(self.config, self.device)
        model.load_state_dict(state_dict, strict=False)
        logger.info('Loaded model from combine')
        return model

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, grad_accum = conf['num_epochs'], conf['gradient_accumulation_steps']

        model.to(self.device)
        logger.info('Model parameters:')
        for name, param in model.named_parameters():
            logger.info('%s: %s' % (name, tuple(param.shape)))

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info('Tensorboard summary path: %s' % tb_path)

        # Set up data
        examples_train, examples_dev, examples_test = self.data.get_tensor_examples()
        stored_info = self.data.get_stored_info()
        model.stored_info = stored_info

        random.seed(self.seed)
        training_samples_ids = [random.sample(range(len(examples_train)), k=len(examples_train)) for _ in range(epochs)]
        # print(training_samples_ids[:5])

        # Set up seed
        if self.seed:
            util.set_seed(self.seed)

        # Set up optimizer and scheduler
        total_update_steps = len(examples_train) * epochs // grad_accum
        optimizers = self.get_optimizer(model)
        schedulers = self.get_scheduler(optimizers, total_update_steps)

        # Get model parameters for grad clipping
        bert_param, task_param = model.get_params()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(examples_train))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        max_f1_test_set = 0
        max_f1_test_set_at_epoch = -1
        max_anaphor_f1 = 0
        max_f1_at_epoch = -1
        max_anaphor_f1_at_epoch = -1
        start_time = time.time()
        model.zero_grad()

        stats_dir = join(conf['log_dir'], 'stats.txt')
        with open(stats_dir, 'w') as f_stats:
            model.config['stats_dir'] = stats_dir

        for epo in range(epochs):
            # random.shuffle(examples_train)  # Shuffle training set
            # for doc_key, example in examples_train:
            for _idx in training_samples_ids[epo]:
                doc_key, example = examples_train[_idx]
                # print(doc_key)
                # Forward pass
                model.train()
                example_gpu = [d.to(self.device) if d is not None else None for d in example]
                _, loss = model(*example_gpu)

                # print stats
                model.config['print_stats'] = False
                if len(loss_history) % 100 == 0:
                    model.config['print_stats'] = True
                    with open(stats_dir, 'a') as f_stats:
                        f_stats.write(f'Stats for training step #{len(loss_history)}, doc_key = {doc_key}\n')

                # Backward; accumulate gradients and clip by grad norm
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                if conf['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(bert_param, conf['max_grad_norm'])
                    torch.nn.utils.clip_grad_norm_(task_param, conf['max_grad_norm'])
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                    model.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', schedulers[0].get_last_lr()[0], len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Task', schedulers[1].get_last_lr()[-1], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 0 and len(loss_history) % conf['eval_frequency'] == 0:
                        save_this_checkpoint = False
                        self.save_model_checkpoint(model, len(loss_history))
                        f1, _, anaphor_r, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history),
                                                            official=False, conll_path=self.config['conll_eval_path'],
                                                            tb_writer=tb_writer)

                        # f1_test, _, _, _ = self.evaluate(model, examples_test, stored_info, len(loss_history),
                        #                                  official=False, conll_path=self.config['conll_eval_path'],
                        #                                  tb_writer=tb_writer)

                        if f1 > max_f1:
                            max_f1 = f1
                            max_f1_at_epoch = len(loss_history)
                            save_this_checkpoint = True
                            self.best_model_suffix = f'{self.name_suffix}_{len(loss_history)}'
                        # if f1_test > max_f1_test_set:
                        #     max_f1_test_set = f1_test
                        #     max_f1_test_set_at_epoch = len(loss_history)
                        #     save_this_checkpoint = True
                        if anaphor_r > max_anaphor_f1:
                            max_anaphor_f1_at_epoch = len(loss_history)
                            max_anaphor_f1 = anaphor_r
                            # save_this_checkpoint = True
                            # self.best_model_suffix = f'{self.name_suffix}_{len(loss_history)}'

                        # if save_this_checkpoint:
                        #     self.save_model_checkpoint(model, len(loss_history))

                        logger.info('Eval max f1: %.2f' % max_f1)
                        logger.info('Eval max f1 on test set: %.2f' % max_f1_test_set)
                        logger.info('Eval max anaphor f1: %.2f' % max_anaphor_f1)
                        start_time = time.time()

                if 'stop_at_epoch' in self.config and len(loss_history) > self.config['stop_at_epoch']:
                    break

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Evaluate
        save_this_checkpoint = False
        self.save_model_checkpoint(model, len(loss_history))
        f1, _, anaphor_r, _ = self.evaluate(model, examples_dev, stored_info, len(loss_history), official=False,
                                            conll_path=self.config['conll_eval_path'], tb_writer=tb_writer)

        # f1_test, _, _, _ = self.evaluate(model, examples_test, stored_info, len(loss_history),
        #                                  official=False, conll_path=self.config['conll_eval_path'],
        #                                  tb_writer=tb_writer)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_at_epoch = len(loss_history)
            save_this_checkpoint = True
            self.best_model_suffix = f'{self.name_suffix}_{len(loss_history)}'
        # if f1_test > max_f1_test_set:
        #     max_f1_test_set = f1_test
        #     max_f1_test_set_at_epoch = len(loss_history)
        #     save_this_checkpoint = True
        if anaphor_r > max_anaphor_f1:
            max_anaphor_f1_at_epoch = len(loss_history)
            max_anaphor_f1 = anaphor_r
            # save_this_checkpoint = True
            # self.best_model_suffix = f'{self.name_suffix}_{len(loss_history)}'
        # if save_this_checkpoint:
        #     self.save_model_checkpoint(model, len(loss_history))
        logger.info('Final Eval max f1 (epoch %d): %.2f' % (max_f1_at_epoch, max_f1))
        logger.info('Final Eval max f1 on test set (epoch %d): %.2f' % (max_f1_test_set_at_epoch, max_f1_test_set))
        logger.info('Final Eval max anaphor f1 (epoch %d): %.2f' % (max_anaphor_f1_at_epoch, max_anaphor_f1))

        del model

        _, _, files = next(os.walk(self.config['log_dir']))
        files = [fn for fn in files if 'model_{}'.format(self.name_suffix) in fn and 'bin' in fn]
        path_ckpt = join(self.config['log_dir'], files[0])
        state_dict = torch.load(path_ckpt, map_location=torch.device(self.device))
        for fn in files[1:]:
            path_ckpt = join(self.config['log_dir'], fn)
            new_state_dict = torch.load(path_ckpt, map_location=torch.device(self.device))
            for layer in state_dict:
                state_dict[layer] += new_state_dict[layer]
        for layer in state_dict:
            state_dict[layer] /= len(files)
        model_to_evaluate = self.initialize_model_from_state_dict(state_dict)
        self.save_combined_model(model_to_evaluate, max_f1_at_epoch, remove=True)

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, tensor_examples, stored_info, step, official=False, conll_path=None, tb_writer=None):
        logger.info('Step %d: evaluating on %d samples...' % (step, len(tensor_examples)))
        model.to(self.device)
        evaluator = CorefEvaluator()
        mention_evaluator = MentionDetectionEvaluator()
        doc_to_prediction = {}

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            gold_clusters = stored_info['gold'][doc_key]
            tensor_example = tensor_example[:-4]  # Strip out gold
            example_gpu = [d.to(self.device) if d is not None else None for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, candidate_anaphor_mask, top_anaphor_pred_types = model(
                    *example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            predicted_clusters = model.update_evaluator(span_starts, span_ends, antecedent_idx, antecedent_scores,
                                                        gold_clusters, evaluator, candidate_anaphor_mask,
                                                        top_anaphor_pred_types)
            _ = model.update_evaluator_anaphor_detection(span_starts, span_ends, antecedent_idx, antecedent_scores,
                                                         gold_clusters, mention_evaluator, candidate_anaphor_mask,
                                                         top_anaphor_pred_types)
            doc_to_prediction[doc_key] = predicted_clusters

            # print('eva', len(predicted_clusters))

            # import ipdb
            # ipdb.set_trace()

        p, r, f = evaluator.get_prf()
        metrics = {'Eval_Avg_Precision': p * 100, 'Eval_Avg_Recall': r * 100, 'Eval_Avg_F1': f * 100}
        for name, score in metrics.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        (p_men, r_men, f1_men), (p_sing, r_sing, f1_sing), (
            p_non_sing, r_non_sing, f1_non_sing) = mention_evaluator.get_prf()
        metrics_men = {'Precision_overall': p_men * 100, 'Recall_overall': r_men * 100, 'F1_overall': f1_men * 100,
                       'Precision_singletons': p_sing * 100, 'Recall_singletons': r_sing * 100,
                       'F1_singletons': f1_sing * 100,
                       'Precision_non_singletons': p_non_sing * 100, 'Recall_non_singletons': r_non_sing * 100,
                       'F1_non_singletons': f1_non_sing * 100}
        for name, score in metrics_men.items():
            logger.info('%s: %.2f' % (name, score))
            if tb_writer:
                tb_writer.add_scalar(name, score, step)

        if official:
            conll_results = conll.evaluate_conll(conll_path, doc_to_prediction, stored_info['subtoken_maps'])
            official_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            logger.info('Official avg F1: %.4f' % official_f1)

        if self.config['constraint_antecedent_utter'] in ['constraint_antecedent_utter',
                                                          'use_predicted_type_as_constraint', 'no_type_constraint']:
            return f * 100, metrics, f1_men * 100, metrics_men
        else:
            raise NotImplementedError

    def predict(self, model, tensor_examples):
        logger.info('Predicting %d samples...' % len(tensor_examples))
        model.to(self.device)
        predicted_spans, predicted_antecedents, predicted_clusters = [], [], []

        model.eval()
        for i, (doc_key, tensor_example) in enumerate(tensor_examples):
            print(doc_key)
            tensor_example = tensor_example[:-4]  # strip out gold
            example_gpu = [d.to(self.device) if d is not None else None for d in tensor_example]
            with torch.no_grad():
                _, _, _, span_starts, span_ends, antecedent_idx, antecedent_scores, candidate_anaphor_mask, top_anaphor_pred_types = model(
                    *example_gpu)
            span_starts, span_ends = span_starts.tolist(), span_ends.tolist()
            antecedent_idx, antecedent_scores = antecedent_idx.tolist(), antecedent_scores.tolist()
            clusters, mention_to_cluster_id, antecedents = model.get_predicted_clusters(span_starts, span_ends,
                                                                                        antecedent_idx,
                                                                                        antecedent_scores,
                                                                                        candidate_anaphor_mask,
                                                                                        top_anaphor_pred_types)

            spans = [(span_start, span_end) for span_start, span_end in zip(span_starts, span_ends)]
            predicted_spans.append(spans)
            predicted_antecedents.append(antecedents)
            predicted_clusters.append(clusters)

        return predicted_clusters, predicted_spans, predicted_antecedents

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_param, task_param = model.get_params(named=True)
        grouped_bert_param = [
            {
                'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
                'lr': self.config['bert_learning_rate'],
                'weight_decay': 0.0
            }
        ]
        optimizers = [
            AdamW(grouped_bert_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps']),
            Adam(model.get_params()[1], lr=self.config['task_learning_rate'], eps=self.config['adam_eps'],
                 weight_decay=0)
        ]
        return optimizers
        # grouped_parameters = [
        #     {
        #         'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['bert_learning_rate'],
        #         'weight_decay': 0.0
        #     }, {
        #         'params': [p for n, p in task_param if not any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': self.config['adam_weight_decay']
        #     }, {
        #         'params': [p for n, p in task_param if any(nd in n for nd in no_decay)],
        #         'lr': self.config['task_learning_rate'],
        #         'weight_decay': 0.0
        #     }
        # ]
        # optimizer = AdamW(grouped_parameters, lr=self.config['task_learning_rate'], eps=self.config['adam_eps'])
        # return optimizer

    def get_scheduler(self, optimizers, total_update_steps):
        # Only warm up bert lr
        warmup_steps = int(total_update_steps * self.config['warmup_ratio'])

        def lr_lambda_bert(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps - warmup_steps))
            )

        def lr_lambda_task(current_step):
            return max(0.0, float(total_update_steps - current_step) / float(max(1, total_update_steps)))

        schedulers = [
            LambdaLR(optimizers[0], lr_lambda_bert),
            LambdaLR(optimizers[1], lr_lambda_task)
        ]
        return schedulers
        # return LambdaLR(optimizer, [lr_lambda_bert, lr_lambda_bert, lr_lambda_task, lr_lambda_task])

    def save_model_checkpoint(self, model, step, remove_prev=False):
        if self.config['debug']:
            return
        if remove_prev:
            _, _, files = next(os.walk(self.config['log_dir']))
            files = [fn for fn in files if 'model_{}'.format(self.name_suffix) in fn and 'bin' in fn]
            files.sort(key=lambda x: int(x[:-4].split('_')[3]), reverse=True)
            for fn in files:
                epoch = int(fn[:-4].split('_')[-1])
                os.remove(join(self.config['log_dir'], fn))
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_{step}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def save_combined_model(self, model, best_model_on_dev, remove=False):
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}_combine.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)
        if remove:
            _, _, files = next(os.walk(self.config['log_dir']))
            files = [fn for fn in files if f'model_{self.name_suffix}' in fn and 'bin' in fn]
            # files.sort(key=lambda x: int(x[:-4].split('_')[3]), reverse=True)
            for fn in files:
                if 'combine' not in fn and str(best_model_on_dev) not in fn.strip('.bin').split('_'):
                    os.remove(join(self.config['log_dir'], fn))

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)


class Evaluator:
    def __init__(self, runner):
        self.runner = runner
        self.custom_name = None

    def conll_score(self):
        if self.runner.best_model_suffix is not None:
            model_to_evaluate = self.runner.initialize_model(self.runner.best_model_suffix)
        else:
            _, _, files = next(os.walk(self.runner.config['log_dir']))
            files = [fn for fn in files if fn.startswith('model_') and 'bin' in fn]
            # files = [fn for fn in files if 'model_{}'.format(self.runner.name_suffix) in fn and 'bin' in fn]
            files.sort(key=lambda x: int(x[:-4].split('_')[3]), reverse=True)
            fn = files[0]

            model_to_evaluate = self.runner.initialize_model(fn[6:-4])

        data_processor = self.runner.data
        jsonlines_path = join(self.runner.config['data_dir'], self.runner.config['test_dir'])
        model_epoch = str(self.runner.best_model_suffix).split('_')[-1]
        test_fn = ''
        if 'ami' in str(self.runner.config['test_dir']).lower():
            test_fn = 'ami'
        elif 'light' in str(self.runner.config['test_dir']).lower():
            test_fn = 'light'
        elif 'pers' in str(self.runner.config['test_dir']).lower():
            test_fn = 'pers'
        elif 'swbd' in str(self.runner.config['test_dir']).lower() or 'switchboard' in str(
                self.runner.config['test_dir']).lower():
            test_fn = 'swbd'
        else:
            print(self.runner.config['test_dir'])
            raise ValueError
        if 'dev' in str(self.runner.config['test_dir']).lower():
            test_fn += '_dev'
        elif 'test' in str(self.runner.config['test_dir']).lower():
            test_fn += '_test'
        else:
            print(self.runner.config['test_dir'])
            raise ValueError
        runner_name = self.runner.name
        if self.custom_name is not None:
            runner_name = runner_name.replace(self.custom_name[0], self.custom_name[1])

        output_path = join(self.runner.config['data_dir'],
                           'output_{}_{}_{}_{}.jsonlines'.format(runner_name, model_epoch, self.runner.name_suffix,
                                                                 test_fn))

        # Input from file
        with open(jsonlines_path, 'r') as f:
            lines = f.readlines()
        with open(output_path, 'w') as f:
            pass
        docs = [json.loads(line) for line in lines]
        for doc in docs:
            tensor_examples, stored_info = data_processor.get_tensor_examples_from_custom_input([doc])
            model_to_evaluate.stored_info = stored_info
            predicted_clusters, _, _ = self.runner.predict(model_to_evaluate, tensor_examples)

            with open(output_path, 'a') as f:
                doc['predicted_clusters_dd'] = predicted_clusters[0]
                # doc['clusters'] = predicted_clusters[i]  # MODIFIED
                f.write(json.dumps(doc))
                f.write('\n')  # MODIFIED
        print(f'Saved prediction in {output_path}')

        json_path = output_path
        conll_path = json_path[:-len('.jsonlines')] + '.CONLLUA'
        helper.convert_coref_json_to_ua(json_path, conll_path, 'anaphor', MODEL="coref-hoi", dd=True)
        print(f'Converted prediction to CONLLUA in {conll_path}')

        if not self.runner.config['prediction_only']:
            gold_conll_path = join(self.runner.config['data_dir'], self.runner.config['gold_conll'])
            import subprocess
            scores = subprocess.check_output(
                ['python', 'ua-scorer.py', gold_conll_path, conll_path, 'evaluate_discourse_deixis'],
                universal_newlines=True)
            for sc in scores.split('\n'):
                logger.info(sc)


def batch_evaluate(gpu_id, seed):
    test_files = [
        None,  # add your test sets here
    ]

    configs = [
        (None, None, None),  # When running batch eval, you need to change this to (config_name, checkpoint_filename, randome_seed)
    ]

    for cfg in configs:
        if len(cfg) == 2:
            config_name, model_to_eval = cfg
            seed_to_eval = seed
        else:
            config_name, model_to_eval, seed_to_eval = cfg
        runner = Runner(config_name.rstrip('/'), gpu_id, seed=seed_to_eval,
                        best_model_suffix=model_to_eval.lstrip('model_').rstrip('.bin/'))
        for test_fn in test_files:
            if ('ami' in config_name.lower() and 'ami' in test_fn.lower()) or \
                    ('light' in config_name.lower() and 'light' in test_fn.lower()) or \
                    ('swbd' in config_name.lower() and 'swbd' in test_fn.lower()) or \
                    ('swbd' in config_name.lower() and 'switchboard' in test_fn.lower()) or \
                    ('pers' in config_name.lower() and 'pers' in test_fn.lower()) or \
                    ('together' in config_name.lower()):
                runner.config['prediction_only'] = True
                # runner.config['use_dummy_antecedent'] = False
                # runner.config['do_type_prediction'] = False
                runner.set_test_file(test_fn)
                evaluator = Evaluator(runner)
                # evaluator.custom_name = ('AllFeat', f'HasNull_SentDist10')
                # evaluator.custom_name = ('AllFeat', f'AllFeatFilteringSentDist{_}')
                # evaluator.custom_name = ('Utterance10', f'Utterance10Seed113')
                evaluator.conll_score()


if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])
    else:
        seed = 11
    if config_name == 'batch':
        batch_evaluate(gpu_id, seed)
    else:
        runner = Runner(config_name, gpu_id, seed=seed)
        model = runner.initialize_model()

        if runner.config['training_phase']:
            runner.train(model)
