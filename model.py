import torch
import torch.nn as nn
from transformers import BertModel
import util
import logging
from collections import Iterable
import numpy as np
import torch.nn.init as init
import higher_order as ho
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class CorefModel(nn.Module):
    def __init__(self, config, device, num_genres=None):
        super().__init__()
        self.config = config
        self.device = device

        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']
        assert config['loss_type'] in ['marginalized', 'hinge']
        if config['coref_depth'] > 1 or config['higher_order'] == 'cluster_merging':
            assert config['fine_grained']  # Higher-order is in slow fine-grained scoring

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        self.bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])

        self.bert_emb_size = self.bert.config.hidden_size
        self.span_emb_size = self.bert_emb_size * 3
        if config['use_features']:
            self.span_emb_size += config['feature_emb_size']
        self.pair_emb_size = self.span_emb_size * 3
        if config['use_metadata']:
            self.pair_emb_size += 2 * config['feature_emb_size']
        if config['use_features']:
            self.pair_emb_size += config['feature_emb_size']
            if config['use_sentence_distance_as_feature']:
                self.pair_emb_size += config['feature_emb_size']
                if self.config['add_cont_values_to_feat']:
                    self.pair_emb_size += 1
                # two types of sentence distance feature
                if not config['ablation_new_features']:
                    self.pair_emb_size += config['feature_emb_size']
                    if self.config['add_cont_values_to_feat']:
                        self.pair_emb_size += 1
            if config['use_extra_features_based_on_antecedent']:
                self.pair_emb_size += 4 * config['feature_emb_size'] + 1
                # number of words raw
                # number of nouns
                # number of verbs
                # number of adjs
                # content word overlap
                # binary feature: if antecedent has longest number of words
                # binary feature: if antecedent has largest number of content word overlaps
                if config['use_content_word_overlap']:
                    self.pair_emb_size += config['feature_emb_size'] + 1
                if self.config['add_cont_values_to_feat']:
                    self.pair_emb_size += 4
                    if config['use_content_word_overlap']:
                        self.pair_emb_size += 1
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_extra_features_based_on_anaphor']:
            self.pair_emb_size += self.span_emb_size

        self.emb_span_width = self.make_embedding(self.max_span_width) if config['use_features'] else None
        self.emb_span_width_prior = self.make_embedding(self.max_span_width) if config['use_width_prior'] else None
        self.emb_antecedent_distance_prior = self.make_embedding(10) if config['use_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config[
            'use_segment_distance'] else None
        self.emb_top_antecedent_distance = self.make_embedding(10)

        if config['max_sentence_distance_between_anaphor_and_antecedent']:
            self.emb_top_antecedent_sentence_distance = self.make_embedding(
                config['max_sentence_distance_between_anaphor_and_antecedent'] + 1) if config[
                'use_sentence_distance_as_feature'] else None  # different from the above one
            self.emb_top_antecedent_sentence_distance_ignore_empty = self.make_embedding(
                config['max_sentence_distance_between_anaphor_and_antecedent'] + 1) if config[
                                                                                           'use_sentence_distance_as_feature'] and not \
                                                                                           config[
                                                                                               'ablation_new_features'] else None  # different from the above one
        else:
            self.emb_top_antecedent_sentence_distance = self.make_embedding(config['max_span_width']) if config['use_sentence_distance_as_feature'] else None  # different from the above one
            self.emb_top_antecedent_sentence_distance_ignore_empty = self.make_embedding(config['max_span_width']) if config['use_sentence_distance_as_feature'] and not \
                                                                                           config['ablation_new_features'] else None  # different from the above one
        self.emb_content_word_overlap = self.make_embedding(self.max_span_width) if config[
            'use_extra_features_based_on_antecedent'] and config['use_content_word_overlap'] else None
        self.emb_num_of_words_raw = self.make_embedding(self.max_span_width) if config[
            'use_extra_features_based_on_antecedent'] else None
        self.emb_num_of_words_remove_nonsense = self.make_embedding(self.max_span_width) if config[
            'use_extra_features_based_on_antecedent'] else None
        self.emb_num_of_words_NnV_remove_reporting = self.make_embedding(self.max_span_width) if config[
            'use_extra_features_based_on_antecedent'] else None
        self.emb_num_of_nouns = self.make_embedding(self.max_span_width) if config[
            'use_extra_features_based_on_antecedent'] else None
        self.emb_num_of_verbs = self.make_embedding(self.max_span_width) if config[
            'use_extra_features_based_on_antecedent'] else None
        self.emb_num_of_adjs = self.make_embedding(self.max_span_width) if config[
            'use_extra_features_based_on_antecedent'] else None

        self.emb_cluster_size = self.make_embedding(10) if config['higher_order'] == 'cluster_merging' else None

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config[
            'model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                                  output_size=1)
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'],
                                                    [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if \
            config['use_width_prior'] else None
        self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) if config[
            'use_distance_prior'] else None
        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                               output_size=1) if config['fine_grained'] else None

        span_type_ffnn_size = self.span_emb_size

        self.span_type_ffnn = self.make_ffnn(span_type_ffnn_size, [config['ffnn_size']] * config['ffnn_depth'],
                                             output_size=config['num_entity_types']) if config[
            'do_type_prediction'] else None

        self.gate_ffnn = self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size) if config[
                                                                                                          'coref_depth'] > 1 else None
        self.span_attn_ffnn = self.make_ffnn(self.span_emb_size, 0, output_size=1) if config[
                                                                                          'higher_order'] == 'span_clustering' else None
        self.cluster_score_ffnn = self.make_ffnn(3 * self.span_emb_size + config['feature_emb_size'],
                                                 [config['cluster_ffnn_size']] * config['ffnn_depth'], output_size=1) if \
            config['higher_order'] == 'cluster_merging' else None

        self.update_steps = 0  # Internal use for debug
        self.debug = config['debug']

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.config['feature_emb_size'])
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, *input):
        return self.get_predictions_and_loss(*input)

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                                 utterance_starts, utterance_ends, utterance_types,
                                 sentence_starts, sentence_ends,
                                 is_training, num_of_words_raw, num_of_words_remove_nonsense,
                                 num_of_words_NnV_remove_reporting, num_of_nouns, num_of_verbs, num_of_adjs,
                                 content_word_overlap, sentence_distance_ignore_empty, rule_based_prediction,
                                 filtered_by_dependency_parsing,
                                 candidate_dep_label,
                                 candidate_dep_other_word_pos_tag,
                                 candidate_dep_other_word_has_relcl,
                                 candidate_dep_other_word_spans_starts,
                                 candidate_dep_other_word_spans_ends,
                                 gold_starts=None, gold_ends=None, gold_types=None,
                                 gold_mention_cluster_map=None):
        """ Model and input are already on the device """
        device = self.device
        conf = self.config
        entity_type_dict = self.stored_info['entity_type_dict']
        entity_type_id_anaphor = entity_type_dict['anaphor']
        entity_type_id_other = entity_type_dict['other']
        entity_type_id_antecedent = entity_type_dict['utterance']

        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            do_loss = True

        # Get token emb
        mention_doc, _ = self.bert(input_ids, attention_mask=input_mask)  # [num seg, num max tokens, emb size]
        input_mask = input_mask.to(torch.bool)
        mention_doc = mention_doc[input_mask]
        speaker_ids = speaker_ids[input_mask]
        num_words = mention_doc.shape[0]

        # Get candidate span
        sentence_indices = sentence_map  # [num tokens]
        candidate_starts = utterance_starts
        candidate_ends = utterance_ends
        candidate_types = utterance_types
        num_candidates = candidate_starts.shape[0]

        # Get candidate labels
        if do_loss:
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).to(torch.long)
            candidate_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                                            same_span.to(torch.float))
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long),
                                             0)  # [num candidates]; non-gold span has label 0

        # Get span embedding & sent embedding
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        candidate_emb_list = [span_start_emb, span_end_emb]

        sent_start_emb, sent_end_emb = mention_doc[sentence_starts], mention_doc[sentence_ends]
        sent_emb_list = [sent_start_emb, sent_end_emb]
        if conf['use_features']:
            # span
            candidate_width_idx = candidate_ends - candidate_starts
            candidate_width_emb = self.emb_span_width(candidate_width_idx)
            candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
            if conf['use_extra_features_based_on_anaphor']:
                # sent
                sent_width_idx = sentence_ends - sentence_starts
                sent_width_emb = self.emb_span_width(sent_width_idx)
                sent_width_emb = self.dropout(sent_width_emb)
                sent_emb_list.append(sent_width_emb)
        # Use attended head or avg token
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1)
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (
                candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
        # for sent
        sent_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(sentence_starts.shape[0], 1)
        sent_tokens_mask = (sent_tokens >= torch.unsqueeze(sentence_starts, 1)) & (
                sent_tokens <= torch.unsqueeze(sentence_ends, 1))
        if conf['model_heads']:
            token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)
        else:
            token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
        candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        candidate_emb_list.append(head_attn_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=1)  # [num candidates, new emb size]

        if conf['use_extra_features_based_on_anaphor']:
            # sent
            sent_tokens_attn_raw = torch.log(sent_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
            sent_tokens_attn = nn.functional.softmax(sent_tokens_attn_raw, dim=1)
            sent_head_attn_emb = torch.matmul(sent_tokens_attn, mention_doc)
            sent_emb_list.append(sent_head_attn_emb)
            sent_span_emb = torch.cat(sent_emb_list, dim=1)  # [num candidates, new emb size]

        # Get span score
        candidate_mention_scores = torch.squeeze(self.span_emb_score_ffnn(candidate_span_emb), 1)
        if conf['use_width_prior']:
            width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
            candidate_width_score = width_score[candidate_width_idx]
            candidate_mention_scores += candidate_width_score

        # get span type score
        if conf['do_type_prediction']:
            emb_for_type_prediction_list = [candidate_span_emb]
            # some code was removed here so the lines above and below look ugly
            emb_for_type_prediction = torch.cat(emb_for_type_prediction_list, dim=1)
            candidate_mention_type_raw_scores = torch.squeeze(self.span_type_ffnn(emb_for_type_prediction), 1)
            candidate_mention_type_scores, candidate_mention_type_ids = torch.topk(
                candidate_mention_type_raw_scores, 1)

        # Extract top spans
        candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
        if conf['use_utterace_span_from_file'] == 'Both':
            if conf['top_span_ratio'] >= 1:
                num_top_spans = len(candidate_starts)
            else:
                num_top_spans = int(
                    min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words, len(candidate_starts)))
            selected_idx_cpu = sorted(candidate_idx_sorted_by_score[:num_top_spans],
                                      key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        else:
            candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
            num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words))
            selected_idx_cpu = self._extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu,
                                                       candidate_ends_cpu, num_top_spans)
        assert len(selected_idx_cpu) == num_top_spans
        selected_idx = torch.tensor(selected_idx_cpu, device=device)
        top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]
        top_span_emb = candidate_span_emb[selected_idx]
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None
        top_span_mention_scores = candidate_mention_scores[selected_idx]
        top_span_types = candidate_types[selected_idx]
        anaphor_mask = (top_span_types == entity_type_id_anaphor) | (top_span_types == entity_type_id_other)
        candidate_antecedent_mask = (top_span_types == entity_type_id_antecedent)

        top_anaphor_pred_types = None
        if conf['do_type_prediction']:
            top_anaphor_gold_types = top_span_types[
                anaphor_mask] if do_loss else None  # [num_top_anaphors] for loss calculation
            top_anaphor_pred_types = candidate_mention_type_ids[selected_idx][anaphor_mask].squeeze(
                1)  # [num_top_spans]

        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        end_sent_idx = sentence_map[top_span_ends]
        start_sent_idx = sentence_map[top_span_starts]
        antecedent_sentence_distance = torch.abs(torch.unsqueeze(start_sent_idx, 1) - torch.unsqueeze(end_sent_idx, 0))
        antecedent_mask = (antecedent_offsets >= 1)
        if conf['max_sentence_distance_between_anaphor_and_antecedent'] > 0:
            antecedent_mask &= (antecedent_sentence_distance < conf[
                'max_sentence_distance_between_anaphor_and_antecedent'])
            antecedent_sentence_distance[
                antecedent_sentence_distance >= conf['max_sentence_distance_between_anaphor_and_antecedent']] = conf[
                'max_sentence_distance_between_anaphor_and_antecedent']
            if conf['use_sentence_distance_as_feature']:
                sentence_distance_ignore_empty[
                    sentence_distance_ignore_empty >= conf['max_sentence_distance_between_anaphor_and_antecedent']] = conf[
                    'max_sentence_distance_between_anaphor_and_antecedent']
        else:
            antecedent_sentence_distance[antecedent_sentence_distance >= conf['max_span_width']] = conf['max_span_width'] - 1
            if conf['use_sentence_distance_as_feature']:
                sentence_distance_ignore_empty[sentence_distance_ignore_empty >= conf['max_span_width']] = conf['max_span_width'] - 1

        # if conf['use_dep_parsing']:
        #     antecedent_mask &= filtered_by_dependency_parsing[selected_idx]

        if conf['constraint_antecedent_utter'] == 'constraint_antecedent_utter':
            antecedent_mask &= (torch.unsqueeze(anaphor_mask, 1) & (candidate_antecedent_mask))
        elif conf['constraint_antecedent_utter'] == 'use_predicted_type_as_constraint':
            assert conf['do_type_prediction']
            constraint_anaphor_mask = torch.zeros(top_span_types.shape, dtype=torch.bool,
                                                  device=self.device)
            constraint_anaphor_mask[anaphor_mask] = (top_anaphor_pred_types == entity_type_id_anaphor)
            constraint_utterance_mask = (top_span_types == entity_type_id_antecedent)
            type_mask = torch.unsqueeze(constraint_anaphor_mask, 1) & torch.unsqueeze(constraint_utterance_mask, 0)
            antecedent_mask &= type_mask
        else:
            anaphor_mask = None

        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(
            top_span_mention_scores, 0)
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))

        # import ipdb
        # ipdb.set_trace()

        if conf['use_extra_features_based_on_anaphor']:
            # dot product of the sentence emb of the sentence that an anaphor is in and every antecedent
            ana_sent_id = sentence_map[top_span_starts[anaphor_mask]]
            ana_sent_emb = sent_span_emb[ana_sent_id]
            ant_span_emb = torch.transpose(top_span_emb, 0, 1)
            extra_pairwise_score = torch.matmul(ana_sent_emb, ant_span_emb)
            pairwise_fast_scores[anaphor_mask] += extra_pairwise_score

        if conf['use_distance_prior']:
            distance_score = torch.squeeze(
                self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = util.bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_idx,
                                                device)  # [num top spans, max top antecedents]
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow mention ranking
        if conf['fine_grained']:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents,
                                                                                     1)
            if conf['use_segment_distance']:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[top_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0,
                                                          self.config['max_training_sentences'] - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                top_antecedent_distance = util.bucket_distance(top_antecedent_offsets)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)
                if self.config['use_sentence_distance_as_feature']:
                    assert (conf['max_sentence_distance_between_anaphor_and_antecedent'] == 0) or (torch.all(antecedent_sentence_distance >= 0) and torch.all(
                        antecedent_sentence_distance <= conf['max_sentence_distance_between_anaphor_and_antecedent']))
                    assert (conf['max_sentence_distance_between_anaphor_and_antecedent'] == 0) or (torch.all(sentence_distance_ignore_empty >= 0) and torch.all(
                        sentence_distance_ignore_empty <= conf['max_sentence_distance_between_anaphor_and_antecedent']))

                    top_antecedent_sentence_distance = util.batch_select(antecedent_sentence_distance,
                                                                         top_antecedent_idx, device)
                    top_antecedent_sentence_distance_emb = self.emb_top_antecedent_sentence_distance(
                        top_antecedent_sentence_distance)
                    if not self.config['ablation_new_features']:
                        assert torch.all(start_sent_idx >= 0) and torch.all(
                            end_sent_idx < len(sentence_distance_ignore_empty))

                        sentence_distance_ignore_empty = sentence_distance_ignore_empty[start_sent_idx, :][:,
                                                         end_sent_idx]
                        top_antecedent_sentence_distance_ignore_empty = util.batch_select(
                            sentence_distance_ignore_empty,
                            top_antecedent_idx, device)
                        top_antecedent_sentence_distance_ignore_empty_emb = self.emb_top_antecedent_sentence_distance_ignore_empty(
                            top_antecedent_sentence_distance_ignore_empty)

            for depth in range(conf['coref_depth']):
                top_antecedent_emb = top_span_emb[top_antecedent_idx]  # [num top spans, max top antecedents, emb size]
                feature_list = []
                if conf['use_metadata']:  # speaker, genre
                    feature_list.append(same_speaker_emb)
                    feature_list.append(genre_emb)
                if conf['use_segment_distance']:
                    feature_list.append(seg_distance_emb)
                if conf['use_features']:  # Antecedent distance
                    feature_list.append(top_antecedent_distance_emb)
                    if self.config['use_sentence_distance_as_feature']:
                        feature_list.append(top_antecedent_sentence_distance_emb)
                        if not self.config['ablation_new_features']:
                            feature_list.append(top_antecedent_sentence_distance_ignore_empty_emb)
                        if self.config['add_cont_values_to_feat']:
                            feature_list.append(top_antecedent_sentence_distance.unsqueeze(2))
                            if not self.config['ablation_new_features']:
                                feature_list.append(top_antecedent_sentence_distance_ignore_empty.unsqueeze(2))

                    if self.config['use_extra_features_based_on_antecedent']:
                        safe_top_span_starts = top_span_starts - 1
                        safe_top_span_starts[safe_top_span_starts < 0] = 0
                        antecedent_num_of_word_raw = num_of_words_raw[top_span_ends] - num_of_words_raw[
                            safe_top_span_starts]
                        # antecedent_num_of_word_remove_nonsense = num_of_words_remove_nonsense[top_span_ends] - \
                        #                                          num_of_words_remove_nonsense[safe_top_span_starts]
                        # antecedent_num_of_word_NnV_remove_reporting = num_of_words_NnV_remove_reporting[top_span_ends] - \
                        #                                               num_of_words_NnV_remove_reporting[
                        #                                                   safe_top_span_starts]
                        antecedent_num_of_nouns = num_of_nouns[top_span_ends] - num_of_nouns[safe_top_span_starts]
                        antecedent_num_of_verbs = num_of_verbs[top_span_ends] - num_of_verbs[safe_top_span_starts]
                        antecedent_num_of_adjs = num_of_adjs[top_span_ends] - num_of_adjs[safe_top_span_starts]

                        antecedent_num_of_word_raw = antecedent_num_of_word_raw.repeat(
                            antecedent_num_of_word_raw.shape[0], 1)
                        # antecedent_num_of_word_remove_nonsense = antecedent_num_of_word_remove_nonsense.repeat(
                        #     antecedent_num_of_word_remove_nonsense.shape[0], 1)
                        # antecedent_num_of_word_NnV_remove_reporting = antecedent_num_of_word_NnV_remove_reporting.repeat(
                        #     antecedent_num_of_word_NnV_remove_reporting.shape[0], 1)
                        antecedent_num_of_nouns = antecedent_num_of_nouns.repeat(antecedent_num_of_nouns.shape[0], 1)
                        antecedent_num_of_verbs = antecedent_num_of_verbs.repeat(antecedent_num_of_verbs.shape[0], 1)
                        antecedent_num_of_adjs = antecedent_num_of_adjs.repeat(antecedent_num_of_adjs.shape[0], 1)

                        # print(len(top_span_starts_cpu))

                        if conf['use_utterace_span_from_file'] == 'Both':
                            antecedent_num_of_word_raw_cpu = antecedent_num_of_word_raw.to('cpu')
                            # antecedent_num_of_word_remove_nonsense_cpu = antecedent_num_of_word_remove_nonsense.to('cpu')
                            # antecedent_num_of_word_NnV_remove_reporting_cpu = antecedent_num_of_word_NnV_remove_reporting.to(
                            #     'cpu')
                            antecedent_num_of_nouns_cpu = antecedent_num_of_nouns.to('cpu')
                            antecedent_num_of_verbs_cpu = antecedent_num_of_verbs.to('cpu')
                            antecedent_num_of_adjs_cpu = antecedent_num_of_adjs.to('cpu')
                            sentence_map_cpu = sentence_map.to('cpu')

                            top_span_starts_cpu = top_span_starts.to('cpu')
                            top_span_ends_cpu = top_span_ends.to('cpu')
                            # sentence_starts_cpu = sentence_starts.cpu()
                            # start_sent_idx_cpu = start_sent_idx.cpu()
                            # top_antecedent_idx_cpu = top_antecedent_idx.cpu()
                            num_of_words_raw_cpu = num_of_words_raw.to('cpu')
                            # num_of_words_remove_nonsense_cpu = num_of_words_remove_nonsense.cpu()
                            # num_of_words_NnV_remove_reporting_cpu = num_of_words_NnV_remove_reporting.cpu()
                            num_of_nouns_cpu = num_of_nouns.to('cpu')
                            num_of_verbs_cpu = num_of_verbs.to('cpu')
                            num_of_adjs_cpu = num_of_adjs.to('cpu')

                            for idx_ana, (st, ed) in enumerate(zip(top_span_starts_cpu, top_span_ends_cpu)):
                                for _ in range(top_span_starts_cpu.shape[0]):
                                    # for _, idx_ant in enumerate(top_antecedent_idx_cpu[idx_ana]):
                                    ant_st = top_span_starts_cpu[_]
                                    # number of words
                                    if _ > idx_ana:
                                        break
                                    if sentence_map_cpu[st] == sentence_map_cpu[ant_st]:
                                        safe_st = max(st - 1, 0)
                                        safe_ant_st = max(ant_st - 1, 0)
                                        antecedent_num_of_word_raw_cpu[idx_ana, _] = num_of_words_raw_cpu[safe_st] - \
                                                                                     num_of_words_raw_cpu[safe_ant_st]
                                        # antecedent_num_of_word_remove_nonsense_cpu[idx_ana, _] = \
                                        # num_of_words_remove_nonsense_cpu[safe_st] - num_of_words_remove_nonsense_cpu[safe_ant_st]
                                        # antecedent_num_of_word_NnV_remove_reporting_cpu[idx_ana, _] = \
                                        # num_of_words_NnV_remove_reporting_cpu[safe_st] - num_of_words_NnV_remove_reporting_cpu[
                                        #     safe_ant_st]
                                        antecedent_num_of_nouns_cpu[idx_ana, _] = num_of_nouns_cpu[safe_st] - \
                                                                                  num_of_nouns_cpu[
                                                                                      safe_ant_st]
                                        antecedent_num_of_verbs_cpu[idx_ana, _] = num_of_verbs_cpu[safe_st] - \
                                                                                  num_of_verbs_cpu[
                                                                                      safe_ant_st]
                                        antecedent_num_of_adjs_cpu[idx_ana, _] = num_of_adjs_cpu[safe_st] - num_of_adjs_cpu[
                                            safe_ant_st]

                            antecedent_num_of_word_raw_cpu = torch.tril(antecedent_num_of_word_raw_cpu)
                            # antecedent_num_of_word_remove_nonsense_cpu = torch.tril(
                            #     antecedent_num_of_word_remove_nonsense_cpu)
                            # antecedent_num_of_word_NnV_remove_reporting_cpu = torch.tril(
                            #     antecedent_num_of_word_NnV_remove_reporting_cpu)
                            antecedent_num_of_nouns_cpu = torch.tril(antecedent_num_of_nouns_cpu)
                            antecedent_num_of_verbs_cpu = torch.tril(antecedent_num_of_verbs_cpu)
                            antecedent_num_of_adjs_cpu = torch.tril(antecedent_num_of_adjs_cpu)

                            # number of words raw
                            antecedent_num_of_word_raw = antecedent_num_of_word_raw_cpu.to(device)
                            # number of nouns, verbs, adjs
                            antecedent_num_of_nouns = antecedent_num_of_nouns_cpu.to(device)
                            antecedent_num_of_verbs = antecedent_num_of_verbs_cpu.to(device)
                            antecedent_num_of_adjs = antecedent_num_of_adjs_cpu.to(device)
                        else:
                            same_sentence_mask = sentence_map[top_span_starts] == sentence_map[top_span_starts].unsqueeze(1)
                            # same_sentence_mask.fill_diagonal_(False)
                            safe_top_span_starts = top_span_starts - 1
                            safe_top_span_starts[safe_top_span_starts < 0] = 0
                            # safe_top_span_starts = safe_top_span_starts.repeat(safe_top_span_starts.shape[0], 1)
                            safe_top_span_ends = top_span_ends.repeat(top_span_ends.shape[0], 1)
                            safe_idx = torch.arange(0, top_span_ends.shape[0], device=device).unsqueeze(1).repeat(1, top_span_ends.shape[0])
                            safe_idx = safe_idx[same_sentence_mask]
                            safe_top_span_ends[same_sentence_mask] = safe_top_span_starts[safe_idx]
                            safe_top_span_starts = safe_top_span_starts.repeat(safe_top_span_starts.shape[0], 1)
                            antecedent_num_of_word_raw = num_of_words_raw[safe_top_span_ends] - num_of_words_raw[safe_top_span_starts]
                            antecedent_num_of_nouns = num_of_nouns[safe_top_span_ends] - num_of_nouns[safe_top_span_starts]
                            antecedent_num_of_verbs = num_of_verbs[safe_top_span_ends] - num_of_verbs[safe_top_span_starts]
                            antecedent_num_of_adjs = num_of_adjs[safe_top_span_ends] - num_of_adjs[safe_top_span_starts]

                            antecedent_num_of_word_raw = torch.tril(antecedent_num_of_word_raw)
                            antecedent_num_of_nouns = torch.tril(antecedent_num_of_nouns)
                            antecedent_num_of_verbs = torch.tril(antecedent_num_of_verbs)
                            antecedent_num_of_adjs = torch.tril(antecedent_num_of_adjs)

                        antecedent_num_of_word_raw = util.batch_select(antecedent_num_of_word_raw,
                                                                       top_antecedent_idx, device)
                        antecedent_num_of_word_raw_emb = self.emb_num_of_words_raw(antecedent_num_of_word_raw)
                        feature_list.append(antecedent_num_of_word_raw_emb)
                        # # number of words remove nonsense
                        # antecedent_num_of_word_remove_nonsense = antecedent_num_of_word_remove_nonsense_cpu.to(device)
                        # antecedent_num_of_word_remove_nonsense = util.batch_select(
                        #     antecedent_num_of_word_remove_nonsense,
                        #     top_antecedent_idx, device)
                        # antecedent_num_of_word_remove_nonsense_emb = self.emb_num_of_words_remove_nonsense(
                        #     antecedent_num_of_word_remove_nonsense)
                        # feature_list.append(antecedent_num_of_word_remove_nonsense_emb)
                        # # number of nouns and verbs remove reporting
                        # antecedent_num_of_word_NnV_remove_reporting = antecedent_num_of_word_NnV_remove_reporting_cpu.to(
                        #     device)
                        # antecedent_num_of_word_NnV_remove_reporting = util.batch_select(
                        #     antecedent_num_of_word_NnV_remove_reporting,
                        #     top_antecedent_idx, device)
                        # antecedent_num_of_word_NnV_remove_reporting_emb = self.emb_num_of_words_NnV_remove_reporting(
                        #     antecedent_num_of_word_NnV_remove_reporting)
                        # feature_list.append(antecedent_num_of_word_NnV_remove_reporting_emb)
                        # number of nouns, verbs, adjs
                        antecedent_num_of_nouns = util.batch_select(antecedent_num_of_nouns,
                                                                    top_antecedent_idx, device)
                        antecedent_num_of_verbs = util.batch_select(antecedent_num_of_verbs,
                                                                    top_antecedent_idx, device)
                        antecedent_num_of_adjs = util.batch_select(antecedent_num_of_adjs,
                                                                   top_antecedent_idx, device)
                        antecedent_num_of_nouns_emb = self.emb_num_of_nouns(antecedent_num_of_nouns)
                        antecedent_num_of_verbs_emb = self.emb_num_of_verbs(antecedent_num_of_verbs)
                        antecedent_num_of_adjs_emb = self.emb_num_of_adjs(antecedent_num_of_adjs)
                        feature_list.append(antecedent_num_of_nouns_emb)
                        feature_list.append(antecedent_num_of_verbs_emb)
                        feature_list.append(antecedent_num_of_adjs_emb)
                        # content word overlap
                        if conf['use_content_word_overlap']:
                            content_word_overlap_selected = util.batch_select(content_word_overlap,
                                                                              top_antecedent_idx, device)
                            content_word_overlap_emb = self.emb_content_word_overlap(content_word_overlap_selected)
                            feature_list.append(content_word_overlap_emb)
                        # binary features
                        ## number of words raw
                        max_antecedent_num_of_word_raw, __ = antecedent_num_of_word_raw.max(1)
                        flag_max_antecedent_num_of_word_raw = (
                                antecedent_num_of_word_raw == max_antecedent_num_of_word_raw.unsqueeze(1))
                        feature_list.append(flag_max_antecedent_num_of_word_raw.unsqueeze(2))
                        # ## number of words remove nonsense
                        # max_antecedent_num_of_word_remove_nonsense, __ = antecedent_num_of_word_remove_nonsense.max(1)
                        # flag_max_antecedent_num_of_word_remove_nonsense = (antecedent_num_of_word_remove_nonsense == max_antecedent_num_of_word_remove_nonsense.unsqueeze(1))
                        # feature_list.append(flag_max_antecedent_num_of_word_remove_nonsense.unsqueeze(2))
                        # ## number of words remove reporting
                        # max_antecedent_num_of_word_NnV_remove_reporting, __ = antecedent_num_of_word_NnV_remove_reporting.max(1)
                        # flag_max_antecedent_num_of_word_NnV_remove_reporting = (antecedent_num_of_word_NnV_remove_reporting == max_antecedent_num_of_word_NnV_remove_reporting.unsqueeze(1))
                        # feature_list.append(flag_max_antecedent_num_of_word_NnV_remove_reporting.unsqueeze(2))
                        ## content word overlap
                        if conf['use_content_word_overlap']:
                            max_content_word_overlap, __ = content_word_overlap_selected.max(1)
                            flag_max_content_word_overlap = (
                                                                    content_word_overlap_selected == max_content_word_overlap.unsqueeze(
                                                                1)) & (max_content_word_overlap.unsqueeze(1) != 0)
                            feature_list.append(flag_max_content_word_overlap.unsqueeze(2))
                        if self.config['add_cont_values_to_feat']:
                            feature_list.append(antecedent_num_of_word_raw.unsqueeze(2))
                            feature_list.append(antecedent_num_of_nouns.unsqueeze(2))
                            feature_list.append(antecedent_num_of_verbs.unsqueeze(2))
                            feature_list.append(antecedent_num_of_adjs.unsqueeze(2))
                            if conf['use_content_word_overlap']:
                                feature_list.append(content_word_overlap_selected.unsqueeze(2))

                feature_emb = torch.cat(feature_list, dim=2)
                feature_emb = self.dropout(feature_emb)
                target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1)
                similarity_emb = target_emb * top_antecedent_emb
                if conf['use_extra_features_based_on_anaphor']:
                    # add sentence emb of the sentence that an anaphor is in and every antecedent
                    ana_sent_id = sentence_map[top_span_starts[anaphor_mask]]
                    ana_sent_emb = sent_span_emb[ana_sent_id]
                    extra_sent_emb = torch.zeros(target_emb.shape, device=self.device)
                    extra_sent_emb[anaphor_mask] += ana_sent_emb.unsqueeze(1)
                    pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, extra_sent_emb, feature_emb],
                                         2)
                else:
                    pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
                top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
                if conf['higher_order'] == 'cluster_merging':
                    cluster_merging_scores = ho.cluster_merging(top_span_emb, top_antecedent_idx, top_pairwise_scores,
                                                                self.emb_cluster_size, self.cluster_score_ffnn, None,
                                                                self.dropout,
                                                                device=device, reduce=conf['cluster_reduce'],
                                                                easy_cluster_first=conf['easy_cluster_first'])
                    break
                elif depth != conf['coref_depth'] - 1:
                    if conf['higher_order'] == 'attended_antecedent':
                        refined_span_emb = ho.attended_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                                  device)
                    elif conf['higher_order'] == 'max_antecedent':
                        refined_span_emb = ho.max_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                             device)
                    elif conf['higher_order'] == 'entity_equalization':
                        refined_span_emb = ho.entity_equalization(top_span_emb, top_antecedent_emb, top_antecedent_idx,
                                                                  top_pairwise_scores, device)
                    elif conf['higher_order'] == 'span_clustering':
                        refined_span_emb = ho.span_clustering(top_span_emb, top_antecedent_idx, top_pairwise_scores,
                                                              self.span_attn_ffnn, device)

                    gate = self.gate_ffnn(torch.cat([top_span_emb, refined_span_emb], dim=1))
                    gate = torch.sigmoid(gate)
                    top_span_emb = gate * refined_span_emb + (1 - gate) * top_span_emb  # [num top spans, span emb size]
        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        if conf['pairwise_penalty_coef1'] > 0:
            temp_top_antecedent_sentence_distance = util.batch_select(antecedent_sentence_distance,
                                                                 top_antecedent_idx, device)
            temp_top_antecedent_sentence_distance[temp_top_antecedent_sentence_distance == 0] = 1
            top_pairwise_scores -= conf['pairwise_penalty_coef1'] * temp_top_antecedent_sentence_distance

        if conf['pairwise_penalty_coef2'] > 0:
            num_of_words_remove_nonsense_ed = num_of_words_remove_nonsense[top_span_ends]
            num_of_words_remove_nonsense_st = num_of_words_remove_nonsense[torch.maximum(top_span_starts-1, torch.tensor(0, device=device, dtype=top_span_starts.dtype))]
            num_of_words_remove_nonsense_each_antecendet = (num_of_words_remove_nonsense_ed - num_of_words_remove_nonsense_st) + 1
            top_pairwise_scores -= conf['pairwise_penalty_coef2'] * 1 / num_of_words_remove_nonsense_each_antecendet[top_antecedent_idx]

        if conf['use_dummy_antecedent']:
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)  # [num top spans, max top antecedents + 1]
        else:
            top_antecedent_scores = top_pairwise_scores

        if conf['hard_cons_c3c4'] != 'Neither':
            assert conf['use_dummy_antecedent']

            predicted_as_anaphor_mask = torch.zeros((top_span_starts.shape), dtype=torch.bool, device=device)
            predicted_as_null_mask = torch.zeros((top_span_starts.shape), dtype=torch.bool, device=device)
            candidate_anaphor_is_predicted_as_anaphor = top_anaphor_pred_types == entity_type_id_anaphor
            candidate_anaphor_is_predicted_as_null = top_anaphor_pred_types == entity_type_id_other
            predicted_as_anaphor_mask[anaphor_mask] = candidate_anaphor_is_predicted_as_anaphor
            predicted_as_null_mask[anaphor_mask] = candidate_anaphor_is_predicted_as_null

            if conf['hard_cons_c3c4'] in ['C3', 'Both']:
                top_antecedent_mask[predicted_as_null_mask, :] = False
                top_antecedent_scores[predicted_as_anaphor_mask, 0] = torch.log(torch.tensor(0.0, device=device)).unsqueeze(0)
            if conf['hard_cons_c3c4'] in ['C4', 'Both']:
                top_antecedent_scores[predicted_as_null_mask, 1:] = torch.log(torch.tensor(0.0, device=device)).unsqueeze(0)

        elif conf['pairwise_penalty_coef3'] > 0 or conf['pairwise_penalty_coef4'] > 0:
            assert conf['use_dummy_antecedent']
            predicted_as_anaphor_mask = torch.zeros((top_span_starts.shape), dtype=torch.bool, device=device)
            predicted_as_null_mask = torch.zeros((top_span_starts.shape), dtype=torch.bool, device=device)
            candidate_anaphor_is_predicted_as_anaphor = top_anaphor_pred_types == entity_type_id_anaphor
            candidate_anaphor_is_predicted_as_null = top_anaphor_pred_types == entity_type_id_other
            predicted_as_anaphor_mask[anaphor_mask] = candidate_anaphor_is_predicted_as_anaphor
            predicted_as_null_mask[anaphor_mask] = candidate_anaphor_is_predicted_as_null
            abs_diff = torch.abs(candidate_mention_type_raw_scores[selected_idx][anaphor_mask][:, 0] - candidate_mention_type_raw_scores[selected_idx][anaphor_mask][:, 1])


            if conf['print_stats'] and do_loss:

                # Get gold labels
                top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
                top_antecedent_cluster_ids += (top_antecedent_mask.to(
                    torch.long) - 1) * 100000  # Mask id on invalid antecedents
                same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
                non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
                pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
                if conf['use_dummy_antecedent']:
                    dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
                    top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)
                    # top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
                    #                                   dim=1)
                else:
                    true_antecedent_labels = pairwise_labels.any(dim=1)
                    top_antecedent_gold_labels = pairwise_labels[true_antecedent_labels]

                gold_idx = top_antecedent_gold_labels.nonzero()[:, 1].unsqueeze(1)
                top_antecedent_scores_gold = util.batch_select(top_antecedent_scores, gold_idx, device=device)
                pred_idx = torch.topk(top_antecedent_scores, 1)[1]
                top_antecedent_scores_predicted = torch.topk(top_antecedent_scores, 1)[0]
                with open(conf['stats_dir'], 'a') as f_stats:
                    temp_abs_diff = abs_diff.detach().clone()
                    temp_abs_diff[candidate_anaphor_is_predicted_as_anaphor] = conf['pairwise_penalty_coef3'] * abs_diff[candidate_anaphor_is_predicted_as_anaphor]
                    temp_abs_diff[candidate_anaphor_is_predicted_as_null] = conf['pairwise_penalty_coef4'] * abs_diff[candidate_anaphor_is_predicted_as_null]
                    temp_abs_diff = temp_abs_diff.tolist()

                    top_antecedent_scores_after_deduction = top_antecedent_scores.detach().clone()
                    top_antecedent_scores_after_deduction[predicted_as_anaphor_mask, 0] -= conf['pairwise_penalty_coef3'] * abs_diff[candidate_anaphor_is_predicted_as_anaphor]
                    top_antecedent_scores_after_deduction[predicted_as_null_mask, 1:] -= conf['pairwise_penalty_coef4'] * abs_diff[
                        candidate_anaphor_is_predicted_as_null].unsqueeze(1)
                    top_antecedent_scores_predicted_after_deduction = torch.topk(top_antecedent_scores_after_deduction, 1)[0]
                    top_antecedent_scores_gold_after_deduction = util.batch_select(top_antecedent_scores_after_deduction, gold_idx, device=device)
                    pred_idx_after_deduction = torch.topk(top_antecedent_scores_after_deduction, 1)[1]

                    to_print = torch.cat([pred_idx, pred_idx_after_deduction, gold_idx, top_antecedent_scores_predicted, top_antecedent_scores_gold, top_antecedent_scores_predicted_after_deduction, top_antecedent_scores_gold_after_deduction, top_antecedent_scores_predicted - top_antecedent_scores_gold], dim=1)
                    to_print = to_print[anaphor_mask]
                    for idx, _ in enumerate(to_print.tolist()):
                        predicted_type_to_print = 'anaphor' if candidate_anaphor_is_predicted_as_anaphor[idx] else 'null'
                        gold_type_to_print = 'anaphor' if top_anaphor_gold_types[idx] == entity_type_id_anaphor else 'null'

                        f_stats.write("{}\t{}\t{:.0f}\t{:.0f}\t{:.0f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t".format(predicted_type_to_print, gold_type_to_print, *_))
                        if candidate_anaphor_is_predicted_as_anaphor[idx]:
                            assert not candidate_anaphor_is_predicted_as_null[idx]
                            f_stats.write('deducting {:.2f} from null antecedent'.format(temp_abs_diff[idx]))
                        else:
                            f_stats.write('deducting {:.2f} from non-null antecedent'.format(temp_abs_diff[idx]))
                        f_stats.write('\n')

                    # import ipdb
                    # ipdb.set_trace()

                # import ipdb
                # ipdb.set_trace()

            top_antecedent_scores[predicted_as_anaphor_mask, 0] -= conf['pairwise_penalty_coef3'] * abs_diff[candidate_anaphor_is_predicted_as_anaphor]
            top_antecedent_scores[predicted_as_null_mask, 1:] -= conf['pairwise_penalty_coef4'] * abs_diff[candidate_anaphor_is_predicted_as_null].unsqueeze(1)


        if not do_loss:
            return candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores, anaphor_mask, top_anaphor_pred_types

        # Get gold labels
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        top_antecedent_cluster_ids += (top_antecedent_mask.to(
            torch.long) - 1) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        if conf['use_dummy_antecedent']:
            dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
            top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)
            # top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
            #                                   dim=1)
            if conf['hard_cons_c3c4'] != 'Neither':
                if conf['hard_cons_c3c4'] in ['C3', 'Both']:
                    predicted_as_anaphor_but_gold_is_null_mask = predicted_as_anaphor_mask & ~dummy_antecedent_labels.squeeze(1)
                else:
                    predicted_as_anaphor_but_gold_is_null_mask = predicted_as_anaphor_mask
                if conf['hard_cons_c3c4'] in ['C4', 'Both']:
                    predicted_as_null_but_gold_is_not_null_mask = ~predicted_as_anaphor_mask & dummy_antecedent_labels.squeeze(1)
                else:
                    predicted_as_null_but_gold_is_not_null_mask = ~predicted_as_anaphor_mask
                hard_cons_c3c4_mask = predicted_as_anaphor_but_gold_is_null_mask | predicted_as_null_but_gold_is_not_null_mask
                top_antecedent_scores = top_antecedent_scores[hard_cons_c3c4_mask]
                top_antecedent_gold_labels = top_antecedent_gold_labels[hard_cons_c3c4_mask]
        else:
            true_antecedent_labels = pairwise_labels.any(dim=1)
            top_antecedent_gold_labels = pairwise_labels[true_antecedent_labels]
            # top_antecedent_scores = top_pairwise_scores[true_antecedent_labels]

        # import ipdb
        # ipdb.set_trace()

        # Get loss
        loss = None
        if conf['loss_type'] == 'marginalized':
            if top_antecedent_scores.shape[0]:
                log_marginalized_antecedent_scores = torch.logsumexp(
                    top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
                log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
                _ = log_norm - log_marginalized_antecedent_scores
                loss = torch.sum(_)

        # Add mention loss
        if conf['mention_loss_coef']:
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids == 0]
            loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores))) * conf['mention_loss_coef']
            _ = 1 - torch.sigmoid(non_gold_mention_scores)
            _[torch.isinf(torch.log(_))] += 1e-10
            loss_mention += -torch.sum(torch.log(_)) * conf[
                'mention_loss_coef']

            # if torch.isnan(loss_mention) or torch.isinf(loss_mention):
            #     import ipdb
            #     ipdb.set_trace()

            if loss is None:
                loss = loss_mention
            else:
                loss += loss_mention
            try:
                assert not torch.isinf(loss) and not torch.isnan(loss)
            except:
                import ipdb
                ipdb.set_trace()

        # Add type loss
        if conf['do_type_prediction'] and len(top_anaphor_gold_types):
            gold_mention_type = top_anaphor_gold_types
            pred_mention_scores = candidate_mention_type_raw_scores[selected_idx][anaphor_mask]

            loss_mention_type = nn.CrossEntropyLoss()(pred_mention_scores, gold_mention_type)

            # if torch.isnan(loss_mention_type) or torch.isinf(loss_mention_type):
            #     import ipdb
            #     ipdb.set_trace()

            loss += loss_mention_type * conf['type_loss_coef']
            try:
                assert not torch.isinf(loss) and not torch.isnan(loss)
            except:
                import ipdb
                ipdb.set_trace()

        # Debug
        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------debug step: %d---------' % self.update_steps)
                # logger.info('candidates: %d; antecedents: %d' % (num_candidates, max_top_antecedents))
                logger.info('spans/gold: %d/%d; ratio: %.2f' % (
                    num_top_spans, (top_span_cluster_ids > 0).sum(), (top_span_cluster_ids > 0).sum() / num_top_spans))
                if conf['mention_loss_coef']:
                    logger.info('mention loss: %.4f' % loss_mention)
                if conf['loss_type'] == 'marginalized':
                    logger.info(
                        'norm/gold: %.4f/%.4f' % (torch.sum(log_norm), torch.sum(log_marginalized_antecedent_scores)))
                else:
                    logger.info('loss: %.4f' % loss)
        self.update_steps += 1

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                top_antecedent_idx, top_antecedent_scores, anaphor_mask, top_anaphor_pred_types], loss

    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
        """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx,
                                        key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx

    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores, anaphor_mask, top_anaphor_pred_types,
                                  entity_type_id_anaphor):
        """ CPU list input """
        predicted_antecedents = []
        _ = 0
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1)):
            if self.config['use_dummy_antecedent']:
                if anaphor_mask is not None and not anaphor_mask[i]:
                    predicted_antecedents.append(-1)
                    continue
                idx -= 1
                if top_anaphor_pred_types is not None and top_anaphor_pred_types[_] != entity_type_id_anaphor:
                    predicted_antecedents.append(-1)
                else:
                    if idx < 0:
                        predicted_antecedents.append(-1)
                    else:
                        predicted_antecedents.append(antecedent_idx[i][idx])
                _ += 1
            else:
                if anaphor_mask is not None and not anaphor_mask[i]:
                    predicted_antecedents.append(-1)
                    continue
                if top_anaphor_pred_types is not None and top_anaphor_pred_types[_] != entity_type_id_anaphor:
                    predicted_antecedents.append(-1)
                else:
                    predicted_antecedents.append(antecedent_idx[i][idx])
                _ += 1
        return predicted_antecedents

    def get_predicted_clusters(self, span_starts, span_ends, antecedent_idx, antecedent_scores, anaphor_mask,
                               top_anaphor_pred_types):
        """ CPU list input """

        entity_type_dict = self.stored_info['entity_type_dict']
        entity_type_id_anaphor = entity_type_dict['anaphor']
        entity_type_id_other = entity_type_dict['other']
        entity_type_id_antecedent = entity_type_dict['utterance']

        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores, anaphor_mask,
                                                               top_anaphor_pred_types, entity_type_id_anaphor)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            try:
                assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            except:
                continue
            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]), 'utterance')
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]), 'anaphor')
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

            # Add singletons
            j = 0
            if self.config['predict_singletons']:
                if top_anaphor_pred_types is not None:
                    for i, span_st in enumerate(span_starts):
                        if not anaphor_mask[i]:
                            continue
                        # Check if it exists in a cluster
                        m_type = 'type'
                        if top_anaphor_pred_types[j] == entity_type_id_other:  # not an actual anaphor
                            m_type = 'other'
                            continue
                        elif top_anaphor_pred_types[j] == entity_type_id_antecedent:  # wtf?
                            m_type = 'utterance'
                            print(top_anaphor_pred_types)
                            import ipdb
                            ipdb.set_trace()
                        elif top_anaphor_pred_types[j] == entity_type_id_anaphor:
                            m_type = 'anaphor'
                        else:
                            raise ValueError
                        mention = (int(span_starts[i]), int(span_ends[i]), m_type)
                        mention_cluster_id = mention_to_cluster_id.get(mention, -1)
                        if mention_cluster_id == -1:
                            mention_cluster_id = len(predicted_clusters)
                            predicted_clusters.append([mention])
                            mention_to_cluster_id[mention] = mention_cluster_id
                        j += 1

        predicted_clusters = [tuple(c) for c in predicted_clusters]
        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def update_evaluator(self, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator,
                         candidate_anaphor_mask, top_anaphor_pred_types):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends,
                                                                                   antecedent_idx, antecedent_scores,
                                                                                   candidate_anaphor_mask,
                                                                                   top_anaphor_pred_types)
        predicted_clusters_removed_type = [tuple(tuple((m[0], m[1])) for m in cluster) for cluster in
                                           predicted_clusters]  # modified
        mention_to_predicted = {m: cluster for cluster in predicted_clusters_removed_type for m in cluster}
        gold_clusters_removed_type = [tuple(tuple((m[0], m[1])) for m in cluster) for cluster in
                                      gold_clusters]  # modified
        mention_to_gold = {m: cluster for cluster in gold_clusters_removed_type for m in cluster}
        evaluator.update(predicted_clusters_removed_type, gold_clusters_removed_type, mention_to_predicted,
                         mention_to_gold)
        return predicted_clusters

    def update_evaluator_anaphor_detection(self, span_starts, span_ends, antecedent_idx, antecedent_scores,
                                           gold_clusters, evaluator, candidate_anaphor_mask, top_anaphor_pred_types):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(span_starts, span_ends,
                                                                                   antecedent_idx, antecedent_scores,
                                                                                   candidate_anaphor_mask,
                                                                                   top_anaphor_pred_types)
        predicted_clusters_removed_type = [tuple(tuple((m[0], m[1])) for m in cluster if m[2] == 'anaphor') for cluster
                                           in predicted_clusters]  # modified
        mention_to_predicted = {m: cluster for cluster in predicted_clusters_removed_type for m in cluster}
        gold_clusters_removed_type = [tuple(tuple((m[0], m[1])) for m in cluster if m[2] == 'anaphor') for cluster in
                                      gold_clusters]  # modified
        mention_to_gold = {m: cluster for cluster in gold_clusters_removed_type for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters_removed_type, mention_to_predicted, mention_to_gold)
        return predicted_clusters
