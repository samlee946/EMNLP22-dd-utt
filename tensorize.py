import tqdm

import util
import numpy as np
import random
import os
from os.path import join
import json
import pickle
import logging
import torch

logger = logging.getLogger(__name__)


class CorefDataProcessor:
    def __init__(self, config, language='english'):
        self.config = config
        self.language = language

        self.max_seg_len = config['max_segment_len']
        self.max_training_seg = config['max_training_sentences']
        self.data_dir = config['data_dir']
        self.train_dir = config['train_dir']  # MODIFIED
        self.dev_dir = config['dev_dir']  # MODIFIED
        self.test_dir = config['test_dir']  # MODIFIED

        self.tokenizer = util.get_tokenizer(config['bert_tokenizer_name'])
        self.tensor_samples, self.stored_info = None, None  # For dataset samples; lazy loading

    def get_tensor_examples_from_custom_input(self, samples):
        """ For interactive samples; no caching """
        tensorizer = Tensorizer(self.config, self.tokenizer)
        tensor_samples = [tensorizer.tensorize_example(sample, 2) for sample in samples]
        tensor_samples = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_samples]
        return tensor_samples, tensorizer.stored_info

    def get_tensor_examples(self):
        """ For dataset samples """
        cache_path = self.get_cache_path()
        if self.config['cache_training_set'] and os.path.exists(cache_path):
            # Load cached tensors if exists
            with open(cache_path, 'rb') as f:
                self.tensor_samples, self.stored_info = pickle.load(f)
                logger.info('Loaded tensorized examples from cache')
        else:
            # Generate tensorized samples
            self.tensor_samples = {}
            self.tensor_samples['tst'] = None
            tensorizer = Tensorizer(self.config, self.tokenizer)
            paths = {
                # 'trn': join(self.data_dir, f'train.{self.language}.{self.max_seg_len}.jsonlines'),
                # 'dev': join(self.data_dir, f'dev.{self.language}.{self.max_seg_len}.jsonlines'),
                # 'tst': join(self.data_dir, f'test.{self.language}.{self.max_seg_len}.jsonlines')
                'trn': join(self.data_dir, f'{self.train_dir}'),
                'dev': join(self.data_dir, f'{self.dev_dir}'),
            }
            if self.test_dir != 'none':
                paths['tst'] = join(self.data_dir, f'{self.test_dir}')

            for split, path in paths.items():
                logger.info('Tensorizing examples from %s; results will be cached)' % path)
                training_flag = 0 if (split == 'trn') else 1 if (split == 'dev') else 2
                with open(path, 'r') as f:
                    samples = [json.loads(line) for line in f.readlines()]
                tensor_samples = [tensorizer.tensorize_example(sample, training_flag) for sample in tqdm.tqdm(samples)]
                self.tensor_ = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_samples
                                if len(tensor[-4]) > 0]  # strip out gold
                self.tensor_samples[split] = self.tensor_
                # in tensor_samples]
                # print(len(tensor_samples), len(self.tensor_samples[split]))
            self.stored_info = tensorizer.stored_info
            # Cache tensorized samples
            with open(cache_path, 'wb') as f:
                pickle.dump((self.tensor_samples, self.stored_info), f)
        return self.tensor_samples['trn'], self.tensor_samples['dev'], self.tensor_samples['tst']

    def get_stored_info(self):
        return self.stored_info

    @classmethod
    def convert_to_torch_tensor(cls, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                utterance_starts, utterance_ends, utterance_types, sentence_starts, sentence_ends,
                                is_training,
                                num_of_words_raw, num_of_words_remove_nonsense, num_of_words_NnV_remove_reporting,
                                num_of_nouns, num_of_verbs, num_of_adjs, content_word_overlap,
                                sentence_distance_ignore_empty, rule_based_prediction,
                                filtered_by_dependency_parsing,
                                candidate_dep_label,
                                candidate_dep_other_word_pos_tag,
                                candidate_dep_other_word_has_relcl,
                                candidate_dep_other_word_spans_starts,
                                candidate_dep_other_word_spans_ends,
                                gold_starts, gold_ends, gold_types, gold_mention_cluster_map):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
        sentence_len = torch.tensor(sentence_len, dtype=torch.long)
        genre = torch.tensor(genre, dtype=torch.long)
        sentence_map = torch.tensor(sentence_map, dtype=torch.long)
        utterance_starts = torch.tensor(utterance_starts, dtype=torch.long)
        utterance_ends = torch.tensor(utterance_ends, dtype=torch.long)
        utterance_types = torch.tensor(utterance_types, dtype=torch.long)
        sentence_starts = torch.tensor(sentence_starts, dtype=torch.long)
        sentence_ends = torch.tensor(sentence_ends, dtype=torch.long)
        is_training = torch.tensor(is_training, dtype=torch.long)

        num_of_words_raw = torch.tensor(num_of_words_raw, dtype=torch.long) if num_of_words_raw is not None else None
        num_of_words_remove_nonsense = torch.tensor(num_of_words_remove_nonsense,
                                                    dtype=torch.long) if num_of_words_remove_nonsense is not None else None
        num_of_words_NnV_remove_reporting = torch.tensor(num_of_words_NnV_remove_reporting,
                                                         dtype=torch.long) if num_of_words_NnV_remove_reporting is not None else None
        num_of_nouns = torch.tensor(num_of_nouns, dtype=torch.long) if num_of_nouns is not None else None
        num_of_verbs = torch.tensor(num_of_verbs, dtype=torch.long) if num_of_verbs is not None else None
        num_of_adjs = torch.tensor(num_of_adjs, dtype=torch.long) if num_of_adjs is not None else None
        content_word_overlap = torch.tensor(content_word_overlap,
                                            dtype=torch.long) if content_word_overlap is not None else None

        sentence_distance_ignore_empty = torch.tensor(sentence_distance_ignore_empty,
                                                      dtype=torch.long) if sentence_distance_ignore_empty is not None else None

        rule_based_prediction = torch.tensor(rule_based_prediction,
                                             dtype=torch.bool) if rule_based_prediction is not None else None
        filtered_by_dependency_parsing = torch.tensor(filtered_by_dependency_parsing,
                                                      dtype=torch.bool) if filtered_by_dependency_parsing is not None else None

        candidate_dep_label = torch.tensor(candidate_dep_label,
                                           dtype=torch.long) if candidate_dep_label is not None else None
        candidate_dep_other_word_pos_tag = torch.tensor(candidate_dep_other_word_pos_tag,
                                                        dtype=torch.long) if candidate_dep_other_word_pos_tag is not None else None
        candidate_dep_other_word_has_relcl = torch.tensor(candidate_dep_other_word_has_relcl,
                                                          dtype=torch.long) if candidate_dep_other_word_has_relcl is not None else None
        candidate_dep_other_word_spans_starts = torch.tensor(candidate_dep_other_word_spans_starts,
                                                             dtype=torch.long) if candidate_dep_other_word_spans_starts is not None else None
        candidate_dep_other_word_spans_ends = torch.tensor(candidate_dep_other_word_spans_ends,
                                                           dtype=torch.long) if candidate_dep_other_word_spans_ends is not None else None

        gold_starts = torch.tensor(gold_starts, dtype=torch.long)
        gold_ends = torch.tensor(gold_ends, dtype=torch.long)
        gold_types = torch.tensor(gold_types, dtype=torch.long)
        gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long)
        return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
               utterance_starts, utterance_ends, utterance_types, sentence_starts, sentence_ends, \
               is_training, \
               num_of_words_raw, num_of_words_remove_nonsense, num_of_words_NnV_remove_reporting, \
               num_of_nouns, num_of_verbs, num_of_adjs, \
               content_word_overlap, sentence_distance_ignore_empty, rule_based_prediction, filtered_by_dependency_parsing, \
               candidate_dep_label, \
               candidate_dep_other_word_pos_tag, \
               candidate_dep_other_word_has_relcl, \
               candidate_dep_other_word_spans_starts, \
               candidate_dep_other_word_spans_ends, \
               gold_starts, gold_ends, gold_types, gold_mention_cluster_map,

    def get_cache_path(self):
        # cache_path = join(self.data_dir, f'cached.tensors.{self.language}.{self.max_seg_len}.{self.max_training_seg}.bin')
        cache_path = join(self.data_dir,
                          f'cached.tensors.{self.train_dir}.{self.dev_dir}.{self.max_seg_len}.{self.max_training_seg}.bin')  # MODIFIED
        return cache_path


class Tensorizer:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Will be used in evaluation
        self.stored_info = {}
        self.stored_info['tokens'] = {}  # {doc_key: ...}
        self.stored_info['subtoken_maps'] = {}  # {doc_key: ...}; mapping back to tokens
        self.stored_info['gold'] = {}  # {doc_key: ...}
        self.stored_info['genre_dict'] = {genre: idx for idx, genre in enumerate(config['genres'])}
        with open(config['entity_type_list'], 'r') as f:
            self.stored_info['entity_type_dict'] = {entity_type.strip(): idx for idx, entity_type in
                                                    enumerate(f.readlines())}
        with open(config['pos_tags_list'], 'r') as f:
            self.stored_info['pos_tag_dict'] = {pos_tag.strip(): idx for idx, pos_tag in
                                                    enumerate(f.readlines())}
        with open(config['dep_tags_list'], 'r') as f:
            self.stored_info['dep_tag_dict'] = {dep_tag.strip(): idx for idx, dep_tag in
                                                    enumerate(f.readlines())}
        print(self.stored_info['entity_type_dict'])

        # for debug only
        self.cnt = 0

    def _tensorize_spans(self, spans):
        if len(spans) > 0:
            starts, ends = zip(*spans)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def _tensorize_span_w_labels(self, spans, label_dict):
        if len(spans) > 0:
            starts, ends, labels = zip(*spans)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[label] for label in labels])

    def _get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for speaker in speakers:
            if len(speaker_dict) > self.config['max_num_speakers']:
                pass  # 'break' to limit # speakers
            if speaker not in speaker_dict:
                speaker_dict[speaker] = len(speaker_dict)
        return speaker_dict

    def tensorize_example(self, example, training_flag):
        # Mentions and clusters
        clusters = example['clusters_dd']
        gold_mentions = sorted(tuple(mention) for mention in util.flatten(clusters))
        # print(gold_mentions)
        gold_mention_map = {mention: idx for idx, mention in enumerate(gold_mentions)}
        gold_mention_cluster_map = np.zeros(len(gold_mentions))  # 0: no cluster
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                gold_mention_cluster_map[gold_mention_map[tuple(mention)]] = cluster_id + 1  # modified

        # Speakers
        speakers = example['speakers']
        speaker_dict = self._get_speaker_dict(util.flatten(speakers))

        # Sentences/segments
        sentences = example['sentences']  # Segments
        sentence_map = example['sentence_map']
        sentence_map = np.array(sentence_map)
        num_words = sum([len(s) for s in sentences])
        max_sentence_len = self.config['max_segment_len']
        sentence_len = np.array([len(s) for s in sentences])
        gold_anaphor_dict = {(st, ed, ty): True for (st, ed, ty) in example['gold_anaphors']}

        # these are not used anymore
        candidate_dep_label = None
        candidate_dep_other_word_pos_tag = None
        candidate_dep_other_word_has_relcl = None
        candidate_dep_other_word_spans_starts = None
        candidate_dep_other_word_spans_ends = None

        if self.config['use_utterace_span_from_file'] == 'Both':
            utterance_spans = example['utterance_span']

            assert len(utterance_spans) == len(example['dep_label'])

            utterance_spans = sorted(utterance_spans, key=lambda x: (x[0], -x[1]))

            utterance_starts, utterance_ends, utterance_types = [], [], []

            for idx, (st, ed, ty) in enumerate(utterance_spans):
                utterance_starts.append(st)
                utterance_ends.append(ed)
                assert ed >= st
                assert st >= 0
                assert ty in self.stored_info['entity_type_dict']
                if training_flag != 0:
                    if (st, ed, 'anaphor') in gold_anaphor_dict:
                        utterance_types.append(self.stored_info['entity_type_dict']['anaphor'])
                    else:
                        utterance_types.append(self.stored_info['entity_type_dict'][ty])
                else:
                    utterance_types.append(self.stored_info['entity_type_dict'][ty])


            utterance_starts = np.array(utterance_starts)
            utterance_ends = np.array(utterance_ends)
            utterance_types = np.array(utterance_types)


        elif self.config['use_utterace_span_from_file'] == 'OnlyAntecedent':

            assert not self.config['use_content_word_overlap']

            utterance_starts = np.repeat(np.expand_dims(np.arange(0, num_words), 1),
                                         self.config['max_span_width_generation'], 1)
            utterance_ends = utterance_starts + np.arange(0, self.config['max_span_width_generation'])
            utterance_start_sent_idx = sentence_map[utterance_starts]
            utterance_end_sent_idx = sentence_map[np.minimum(utterance_ends, num_words - 1)]
            utterance_mask = (utterance_ends < num_words) & (utterance_start_sent_idx == utterance_end_sent_idx)
            utterance_starts = utterance_starts[utterance_mask]
            utterance_ends = utterance_ends[utterance_mask]
            utterance_types = np.zeros(utterance_starts.shape)

            utterance_spans = set([(st, ed, ty) for (st, ed, ty) in example['utterance_span']])

            utterance_mask = np.ones(utterance_starts.shape, dtype=np.bool)
            for idx, (st, ed) in enumerate(zip(utterance_starts, utterance_ends)):
                if (st, ed, 'utterance') in utterance_spans:
                    utterance_mask[idx] = False
                elif (st, ed, 'anaphor') in utterance_spans:
                    utterance_types[idx] = self.stored_info['entity_type_dict']['anaphor']
                else:
                    utterance_types[idx] = self.stored_info['entity_type_dict']['other']

            utterance_starts = utterance_starts[utterance_mask]
            utterance_ends = utterance_ends[utterance_mask]
            utterance_types = utterance_types[utterance_mask]

            utterance_starts = list(utterance_starts)
            utterance_ends = list(utterance_ends)
            utterance_types = list(utterance_types)

            for (st, ed, ty) in utterance_spans:
                if ty == 'utterance':
                    utterance_starts.append(st)
                    utterance_ends.append(ed)
                    utterance_types.append(self.stored_info['entity_type_dict']['utterance'])

            utterance_starts, utterance_ends, utterance_types = zip(
                *sorted(zip(utterance_starts, utterance_ends, utterance_types), key=lambda x: (x[0], -x[1])))

            utterance_starts = np.array(utterance_starts)
            utterance_ends = np.array(utterance_ends)
            utterance_types = np.array(utterance_types)

        elif self.config['use_utterace_span_from_file'] == 'OnlyAnaphor':
            utterance_starts = np.repeat(np.expand_dims(np.arange(0, num_words), 1),
                                         self.config['max_span_width_generation'], 1)
            utterance_ends = utterance_starts + np.arange(0, self.config['max_span_width_generation'])
            utterance_start_sent_idx = sentence_map[utterance_starts]
            utterance_end_sent_idx = sentence_map[np.minimum(utterance_ends, num_words - 1)]
            utterance_mask = (utterance_ends < num_words) & (utterance_start_sent_idx == utterance_end_sent_idx)
            utterance_starts = utterance_starts[utterance_mask]
            utterance_ends = utterance_ends[utterance_mask]
            utterance_types = np.zeros(utterance_starts.shape)

            anaphor_span_from_file = set([(st, ed, ty) for (st, ed, ty) in example['utterance_span']])

            for idx in range(utterance_starts.shape[0]):
                st, ed = utterance_starts[idx], utterance_ends[idx]
                if (st, ed, 'anaphor') in anaphor_span_from_file:
                    utterance_types[idx] = 1
                elif (st, ed, 'other') in anaphor_span_from_file:
                    utterance_types[idx] = 0
                else:
                    utterance_types[idx] = 2

        elif self.config['use_utterace_span_from_file'] == 'Neither':
            utterance_starts = np.repeat(np.expand_dims(np.arange(0, num_words), 1),
                                         self.config['max_span_width_generation'], 1)
            utterance_ends = utterance_starts + np.arange(0, self.config['max_span_width_generation'])
            utterance_start_sent_idx = sentence_map[utterance_starts]
            utterance_end_sent_idx = sentence_map[np.minimum(utterance_ends, num_words - 1)]
            utterance_mask = (utterance_ends < num_words) & (utterance_start_sent_idx == utterance_end_sent_idx)
            utterance_starts = utterance_starts[utterance_mask]
            utterance_ends = utterance_ends[utterance_mask]
            utterance_types = np.ones(utterance_starts.shape)
        else:
            logger.info("{} not recognized".format(self.config['use_utterace_span_from_file']))
            raise NotImplementedError

        sentence_starts, sentence_ends = [], []
        for index, i in enumerate(sentence_map):
            if i >= len(sentence_starts):
                sentence_starts.append(index)
                sentence_ends.append(index)
            else:
                sentence_ends[i] = max(sentence_ends[i], index)
        sentence_starts = np.array(sentence_starts)
        sentence_ends = np.array(sentence_ends)

        # Bert input
        input_ids, input_mask, speaker_ids = [], [], []
        for idx, (sent_tokens, sent_speakers) in enumerate(zip(sentences, speakers)):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict[speaker] for speaker in sent_speakers]
            while len(sent_input_ids) < max_sentence_len:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            speaker_ids.append(sent_speaker_ids)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        # Keep info to store
        doc_key = example['doc_key']
        self.stored_info['subtoken_maps'][doc_key] = example.get('subtoken_map', None)
        self.stored_info['gold'][doc_key] = example['clusters_dd']
        # self.stored_info['tokens'][doc_key] = example['tokens']

        # Construct example
        genre = self.stored_info['genre_dict'].get(doc_key[:2], 0)
        # gold_starts, gold_ends = self._tensorize_spans(gold_mentions)
        gold_starts, gold_ends, gold_types = self._tensorize_span_w_labels(gold_mentions,
                                                                           self.stored_info['entity_type_dict'])
        assert len(gold_ends) == len(gold_starts)

        num_of_words_raw = None
        num_of_words_remove_nonsense = None
        num_of_words_NnV_remove_reporting = None
        num_of_nouns = None
        num_of_verbs = None
        num_of_adjs = None
        content_word_overlap = None

        if 'num_of_words_raw' in example:# and self.config['use_extra_features_based_on_antecedent']:
            assert 'num_of_words_remove_non_sense' not in example
            num_of_words_raw = np.array(example['num_of_words_raw'])
            num_of_words_remove_nonsense = np.array(example['num_of_words_remove_non_sense_and_punc'])
            num_of_words_NnV_remove_reporting = np.array(example['num_of_words_NnV_remove_reporting'])
            num_of_nouns = np.array(example['num_of_nouns'])
            num_of_verbs = np.array(example['num_of_verbs'])
            num_of_adjs = np.array(example['num_of_adjs'])
            content_word_overlap = None
            if self.config['use_content_word_overlap']:
                content_word_overlap = np.zeros((len(utterance_starts), len(utterance_starts)))
                for (ana_id, utt_id, num_of_content_word_overlap) in example['num_of_content_word_overlap']:
                    content_word_overlap[ana_id, utt_id] = num_of_content_word_overlap

        sentence_distance_ignore_empty = None
        if self.config['use_sentence_distance_as_feature']:
            try:
                assert 'num_of_words_NnV_remove_reporting' in example
                # assert self.config['max_sentence_distance_between_anaphor_and_antecedent'] > 0
            except:
                print(example['doc_key'])
                print(self.config['train_dir'])
                print(self.config['dev_dir'])
                print(self.config['test_dir'])
                raise AssertionError
            sentence_distance_ignore_empty = np.ones((len(sentence_starts), len(sentence_starts))) * len(
                sentence_starts)

            num_of_words_NnV_remove_reporting = np.array(example['num_of_words_NnV_remove_reporting'])
            # utterance_starts_list = list(utterance_starts)
            # utterance_ends_list = list(utterance_ends)
            # utterance_types_list = list(utterance_types)
            # for ana_id, (st, ed, ty) in enumerate(zip(utterance_starts_list, utterance_ends_list, utterance_types_list)):
            #     if ty == self.stored_info['entity_type_dict']['utterance'] or ana_id == 0:
            #         continue
            #     # print(ana_id, len(utterance_starts))
            #     start_distance = -1
            #     for ant_id in range(ana_id - 1, -1, -1):
            #         utt_st = utterance_starts_list[ant_id]
            #         utt_ed = utterance_ends_list[ant_id]
            #         utt_ty = utterance_types_list[ant_id]
            #         if utt_ty != self.stored_info['entity_type_dict']['utterance']:
            #             continue
            #         elif abs(sentence_map[st] - sentence_map[utt_st]) >= self.config['max_sentence_distance_between_anaphor_and_antecedent'] or \
            #                 start_distance >= self.config['max_sentence_distance_between_anaphor_and_antecedent']:
            #             break
            #         else:
            #             if start_distance == -1:
            #                 start_distance = int(~(sentence_map[st] == sentence_map[utt_st]))
            #             # sentence_distance_ignore_empty[ana_id, ant_id] = start_distance
            #             s_n_o_words = 0 if utt_st == 0 else num_of_words_NnV_remove_reporting[utt_st - 1]
            #             n_o_words = num_of_words_NnV_remove_reporting[utt_ed] - s_n_o_words
            #             if n_o_words != 0:
            #                 start_distance += 1
            for sent_id, (st, ed) in enumerate(zip(sentence_starts, sentence_ends)):
                start_distance = 0
                for ant_sent_id in range(sent_id, -1, -1):
                    utt_st = sentence_starts[ant_sent_id]
                    utt_ed = sentence_ends[ant_sent_id]
                    if abs(sentence_map[st] - sentence_map[utt_st]) >= self.config[
                        'max_sentence_distance_between_anaphor_and_antecedent'] or \
                            start_distance >= self.config['max_sentence_distance_between_anaphor_and_antecedent']:
                        break
                    else:
                        s_n_o_words = 0 if utt_st == 0 else num_of_words_NnV_remove_reporting[utt_st - 1]
                        n_o_words = num_of_words_NnV_remove_reporting[utt_ed] - s_n_o_words
                        sentence_distance_ignore_empty[sent_id, ant_sent_id] = start_distance
                        if n_o_words != 0:
                            start_distance += 1
                        # else:
                        #     print(doc_key, ant_sent_id)


        rule_based_prediction = None
        if 'rule_based_prediction' in example and self.config['use_rule_based_prediction']:
            rule_based_prediction = np.zeros((len(utterance_starts), len(utterance_starts)))
            for (ana_id, utt_id) in example['rule_based_prediction']:
                rule_based_prediction[ana_id, utt_id] = 1

        filtered_by_dependency_parsing = None
        if 'filtered_by_dependency_parsing' in example and (
                self.config['inference_filter_dependency_parsing'] or self.config['use_dep_parsing']):
            filtered_by_dependency_parsing = np.ones((len(utterance_starts), len(utterance_starts)))
            for (ana_id, utt_id) in example['filtered_by_dependency_parsing']:
                filtered_by_dependency_parsing[ana_id, utt_id] = 0

        example_tensor = (
            input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, utterance_starts, utterance_ends,
            utterance_types, sentence_starts, sentence_ends, training_flag,
            num_of_words_raw, num_of_words_remove_nonsense, num_of_words_NnV_remove_reporting, num_of_nouns,
            num_of_verbs, num_of_adjs, content_word_overlap, sentence_distance_ignore_empty, rule_based_prediction,
            filtered_by_dependency_parsing,
            candidate_dep_label,
            candidate_dep_other_word_pos_tag,
            candidate_dep_other_word_has_relcl,
            candidate_dep_other_word_spans_starts,
            candidate_dep_other_word_spans_ends,
            gold_starts, gold_ends, gold_types, gold_mention_cluster_map)

        if training_flag == 0 and len(sentences) > self.config['max_training_sentences']:
            example_tensor = self.truncate_example(*example_tensor)

        return doc_key, example_tensor

    def truncate_example(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, utterance_starts,
                         utterance_ends, utterance_types, sentence_starts, sentence_ends, is_training,
                         num_of_words_raw, num_of_words_remove_nonsense, num_of_words_NnV_remove_reporting,
                         num_of_nouns, num_of_verbs, num_of_adjs, content_word_overlap, sentence_distance_ignore_empty,
                         rule_based_prediction,
                         filtered_by_dependency_parsing,
                         candidate_dep_label,
                         candidate_dep_other_word_pos_tag,
                         candidate_dep_other_word_has_relcl,
                         candidate_dep_other_word_spans_starts,
                         candidate_dep_other_word_spans_ends,
                         gold_starts, gold_ends, gold_types, gold_mention_cluster_map, sentence_offset=None):
        max_sentences = self.config["max_training_sentences"]
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_sentences

        sent_offset = sentence_offset
        if sent_offset is None:
            sent_offset = random.randint(0, num_sentences - max_sentences)
        word_offset = sentence_len[:sent_offset].sum()
        num_words = sentence_len[sent_offset: sent_offset + max_sentences].sum()

        input_ids = input_ids[sent_offset: sent_offset + max_sentences, :]
        input_mask = input_mask[sent_offset: sent_offset + max_sentences, :]
        speaker_ids = speaker_ids[sent_offset: sent_offset + max_sentences, :]
        sentence_len = sentence_len[sent_offset: sent_offset + max_sentences]

        # 11/15 model original
        # sentence_map = sentence_map[word_offset: word_offset + num_words]
        # sentence_map -= sentence_map[0]
        # gold_spans = (gold_ends < word_offset + num_words) & (gold_starts >= word_offset)
        # gold_starts = gold_starts[gold_spans] - word_offset
        # gold_ends = gold_ends[gold_spans] - word_offset
        # gold_types = gold_types[gold_spans]
        #
        # utterance_spans = (utterance_ends < word_offset + num_words) & (utterance_starts >= word_offset)
        # utterance_starts = utterance_starts[utterance_spans] - word_offset
        # utterance_ends = utterance_ends[utterance_spans] - word_offset
        # utterance_types = utterance_types[utterance_spans]
        #
        # sentence_spans = (sentence_ends < word_offset + num_words) & (sentence_starts >= word_offset)
        # sentence_starts = sentence_starts[sentence_spans] - word_offset
        # sentence_ends = sentence_ends[sentence_spans] - word_offset

        # added 2021/12/20
        sentence_map = sentence_map[word_offset: word_offset + num_words]
        sentence_map -= sentence_map[0]
        gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        gold_types = gold_types[gold_spans]

        utterance_spans = (utterance_starts < word_offset + num_words) & (utterance_ends >= word_offset)
        utterance_starts = utterance_starts[utterance_spans] - word_offset
        utterance_ends = utterance_ends[utterance_spans] - word_offset
        utterance_starts[utterance_starts < 0] = 0
        utterance_ends[utterance_ends >= num_words] = num_words - 1
        utterance_types = utterance_types[utterance_spans]

        sentence_spans = (sentence_starts < word_offset + num_words) & (sentence_ends >= word_offset)
        sentence_starts = sentence_starts[sentence_spans] - word_offset
        sentence_ends = sentence_ends[sentence_spans] - word_offset
        sentence_starts[sentence_starts < 0] = 0
        sentence_ends[sentence_ends >= num_words] = num_words - 1

        gold_mention_cluster_map = gold_mention_cluster_map[gold_spans]

        # added 2022/01/07
        if num_of_words_raw is not None:
            num_of_words_raw = num_of_words_raw[word_offset: word_offset + num_words]
            num_of_words_raw -= num_of_words_raw[0]
            num_of_words_remove_nonsense = num_of_words_remove_nonsense[word_offset: word_offset + num_words]
            num_of_words_remove_nonsense -= num_of_words_remove_nonsense[0]
            num_of_words_NnV_remove_reporting = num_of_words_NnV_remove_reporting[word_offset: word_offset + num_words]
            num_of_words_NnV_remove_reporting -= num_of_words_NnV_remove_reporting[0]
            num_of_nouns = num_of_nouns[word_offset: word_offset + num_words]
            num_of_nouns -= num_of_nouns[0]
            num_of_verbs = num_of_verbs[word_offset: word_offset + num_words]
            num_of_verbs -= num_of_verbs[0]
            num_of_adjs = num_of_adjs[word_offset: word_offset + num_words]
            num_of_adjs -= num_of_adjs[0]
            if self.config['use_content_word_overlap']:
                content_word_overlap = content_word_overlap[utterance_spans, :]
                content_word_overlap = content_word_overlap[:, utterance_spans]

        if sentence_distance_ignore_empty is not None:
            sentence_distance_ignore_empty = sentence_distance_ignore_empty[sentence_spans, :]
            sentence_distance_ignore_empty = sentence_distance_ignore_empty[:, sentence_spans]

        if rule_based_prediction is not None:
            rule_based_prediction = rule_based_prediction[utterance_spans, :]
            rule_based_prediction = rule_based_prediction[:, utterance_spans]

        if filtered_by_dependency_parsing is not None:
            filtered_by_dependency_parsing = filtered_by_dependency_parsing[utterance_spans, :]
            filtered_by_dependency_parsing = filtered_by_dependency_parsing[:, utterance_spans]

        if candidate_dep_label is not None:
            candidate_dep_label = candidate_dep_label[utterance_spans]
            candidate_dep_other_word_pos_tag = candidate_dep_other_word_pos_tag[utterance_spans]
            candidate_dep_other_word_has_relcl = candidate_dep_other_word_has_relcl[utterance_spans]
            candidate_dep_other_word_spans_starts = candidate_dep_other_word_spans_starts[utterance_spans] - word_offset
            candidate_dep_other_word_spans_ends = candidate_dep_other_word_spans_ends[utterance_spans] - word_offset

        # return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, [1], [1], [1], \
        #        is_training, gold_starts, gold_ends, gold_types, gold_mention_cluster_map
        return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, utterance_starts, utterance_ends, utterance_types, sentence_starts, sentence_ends, \
               is_training, num_of_words_raw, num_of_words_remove_nonsense, num_of_words_NnV_remove_reporting, num_of_nouns, num_of_verbs, num_of_adjs, content_word_overlap, sentence_distance_ignore_empty, rule_based_prediction, \
               filtered_by_dependency_parsing, \
               candidate_dep_label, \
               candidate_dep_other_word_pos_tag, \
               candidate_dep_other_word_has_relcl, \
               candidate_dep_other_word_spans_starts, \
               candidate_dep_other_word_spans_ends, \
               gold_starts, gold_ends, gold_types, gold_mention_cluster_map
