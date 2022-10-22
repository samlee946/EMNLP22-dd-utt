import json
import os
from os import walk, makedirs
from os.path import isfile, join
import collections
from collections import deque
import sys
import logging
import re
import numpy as np
import random
from transformers import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def flatten(l):
    return [item for sublist in l for item in sublist]


def get_tokenizer(bert_tokenizer_name):
    return BertTokenizer.from_pretrained(bert_tokenizer_name)


def skip_doc(doc_key):
    return False

def normalize_word(word, language):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word


def get_sentence_map(segments, sentence_end):
    assert len(sentence_end) == sum([len(seg) - 2 for seg in segments])  # of subtokens in all segments
    sent_map = []
    sent_idx, subtok_idx = 0, 0
    for segment in segments:
        sent_map.append(sent_idx)  # [CLS]
        for i in range(len(segment) - 2):
            sent_map.append(sent_idx)
            sent_idx += int(sentence_end[subtok_idx])
            subtok_idx += 1
        sent_map.append(sent_idx)  # [SEP]
    return sent_map


class Markable:
    def __init__(self, doc_name, start, end, MIN, is_referring, words,is_split_antecedent=False,split_antecedent_members=set()):
        self.doc_name = doc_name
        self.start = start
        self.end = end
        self.MIN = MIN
        self.is_referring = is_referring
        self.words = words
        self.is_split_antecedent = is_split_antecedent
        self.split_antecedent_members = split_antecedent_members

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # for split-antecedent we check all the members are the same
            if self.is_split_antecedent or other.is_split_antecedent:
                return self.split_antecedent_members == other.split_antecedent_members
            # MIN is only set for the key markables
            elif self.MIN:
                return (self.doc_name == other.doc_name
                        and other.start >= self.start
                        and other.start <= self.MIN[0]
                        and other.end <= self.end
                        and other.end >= self.MIN[1])
            elif other.MIN:
                return (self.doc_name == other.doc_name
                        and self.start >= other.start
                        and self.start <= other.MIN[0]
                        and self.end <= other.end
                        and self.end >= other.MIN[1])
            else:
                return (self.doc_name == other.doc_name
                        and self.start == other.start
                        and self.end == other.end)
        return NotImplemented

    def __neq__(self, other):
        if isinstance(other, self.__class__):
            return self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        if self.is_split_antecedent:
            return hash(frozenset(self.split_antecedent_members))
        return hash(frozenset((self.start, self.end)))

    def __short_str__(self):
        return ('({},{})'.format(self.start,self.end))

    def __str__(self):
        if self.is_split_antecedent:
            return str([cl[0].__short_str__() for cl in self.split_antecedent_members])
        return self.__short_str__()
            # ('DOC: %s SPAN: (%d, %d) String: %r MIN: %s Referring tag: %s'
            #	 % (
            #		 self.doc_name, self.start, self.end, ' '.join(self.words),
            #		 '(%d, %d)' % self.MIN if self.MIN else '',
            #		 self.is_referring))

class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.tokens = []

        # Linear list mapped to subtokens without CLS, SEP
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []  # Only non-none for the first subtoken of each word

        # Linear list mapped to subtokens with CLS, SEP
        self.sentence_map = []

        # Segments (mapped to subtokens with CLS, SEP)
        self.segments = []
        self.segment_subtoken_map = []
        self.segment_info = []  # Only non-none for the first subtoken of each word
        self.speakers = []

        # Doc-level attributes
        self.pronouns = []
        self.clusters = collections.defaultdict(list)  # {cluster_id: [(first_subtok_idx, last_subtok_idx) for each mention]}
        self.coref_stacks = collections.defaultdict(list)

    def finalize(self):
        """ Extract clusters; fill other info e.g. speakers, pronouns """
        # Populate speakers from info
        subtoken_idx = 0
        for seg_info in self.segment_info:
            speakers = []
            for i, subtoken_info in enumerate(seg_info):
                if i == 0 or i == len(seg_info) - 1:
                    speakers.append('[SPL]')
                elif subtoken_info is not None:  # First subtoken of each word
                    speakers.append(subtoken_info[9])
                    # if subtoken_info[4] == 'PRP':  # Uncomment if needed
                    #     self.pronouns.append(subtoken_idx)
                else:
                    speakers.append(speakers[-1])
                subtoken_idx += 1
            self.speakers += [speakers]

        # Populate cluster
        first_subtoken_idx = 0  # Subtoken idx across segments
        subtokens_info = flatten(self.segment_info)
        while first_subtoken_idx < len(subtokens_info):
            subtoken_info = subtokens_info[first_subtoken_idx]
            coref = subtoken_info[-2] if subtoken_info is not None else '-'
            if coref != '-':
                last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1
                for part in coref.split('|'):
                    if part[0] == '(':
                        if part[-1] == ')':
                            cluster_id = int(part[1:-1])
                            self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx))
                        else:
                            cluster_id = int(part[1:])
                            self.coref_stacks[cluster_id].append(first_subtoken_idx)
                    else:
                        cluster_id = int(part[:-1])
                        start = self.coref_stacks[cluster_id].pop()
                        self.clusters[cluster_id].append((start, last_subtoken_idx))
            first_subtoken_idx += 1

        # Merge clusters if any clusters have common mentions
        merged_clusters = []
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often)")
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))

        merged_clusters = [list(cluster) for cluster in merged_clusters]
        all_mentions = flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)

        # Sanity check
        assert len(all_mentions) == len(set(all_mentions))  # Each mention unique
        # Below should have length: # all subtokens with CLS, SEP in all segments
        num_all_seg_tokens = len(flatten(self.segments))
        assert num_all_seg_tokens == len(flatten(self.speakers))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)

        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "ner": [],
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }

class UADocumentState(DocumentState):
    def __init__(self, key):
        self.doc_key = key
        self.tokens = []

        # Linear list mapped to subtokens without CLS, SEP
        self.subtokens = []
        self.subtoken_map = []
        self.token_end = []
        self.sentence_end = []
        self.info = []  # Only non-none for the first subtoken of each word

        # Linear list mapped to subtokens with CLS, SEP
        self.sentence_map = []

        # Segments (mapped to subtokens with CLS, SEP)
        self.segments = []
        self.segment_subtoken_map = []
        self.segment_info = []  # Only non-none for the first subtoken of each word
        self.speakers = []

        # Doc-level attributes
        self.pronouns = []
        self.clusters = collections.defaultdict(list)  # {cluster_id: [(first_subtok_idx, last_subtok_idx) for each mention]}
        self.coref_stacks = []#collections.defaultdict(list)

    def finalize(self):
        """ Extract clusters; fill other info e.g. speakers, pronouns """
        # Populate speakers from info
        subtoken_idx = 0
        for seg_info in self.segment_info:
            speakers = []
            for i, subtoken_info in enumerate(seg_info):
                if i == 0 or i == len(seg_info) - 1:
                    speakers.append('[SPL]')
                elif subtoken_info is not None:  # First subtoken of each word
                    speakers.append(subtoken_info[9])
                    # if subtoken_info[4] == 'PRP':  # Uncomment if needed
                    #     self.pronouns.append(subtoken_idx)
                else:
                    speakers.append(speakers[-1])
                subtoken_idx += 1
            self.speakers += [speakers]

        # Populate cluster
        overlapping_span = {}
        first_subtoken_idx = 0  # Subtoken idx across segments
        subtokens_info = flatten(self.segment_info)
        while first_subtoken_idx < len(subtokens_info):
            subtoken_info = subtokens_info[first_subtoken_idx]
            coref = subtoken_info[10] if subtoken_info is not None else '-'
            semantics = subtoken_info[-2] if subtoken_info is not None else '-'
#             logger.info(semantics)
            if coref != '-' and coref != '' and coref != '_':
                last_subtoken_idx = first_subtoken_idx + subtoken_info[-1] - 1

                parts = coref.split('(')
                parts_semantics = semantics.split('(')
                assert len(parts) == len(parts_semantics)
#                 logger.info('{} {}'.format(parts, subtoken_info[-1]))
                if len(parts[0]):
                    #the close bracket
                    for _ in range(len(parts[0])):
                        cluster_id, start, entity_type = self.coref_stacks.pop()
#                         if (start, last_subtoken_idx) not in overlapping_span:
                        self.clusters[cluster_id].append((start, last_subtoken_idx, entity_type))
#                             overlapping_span[(start, last_subtoken_idx)] = True
                for part, part_semantics in zip(parts[1:], parts_semantics[1:]):
#                     print(coref)
                    entity = part.split("|")[0].split("=")[1].split("-")[0]
                    entity_type = 'entity'
                    if '-Pseudo' in part.split("|")[0].split("=")[1]:
                        entity_type = 'non_refering'
#                         print(entity, entity_type)
#                         logger.info('{} {}'.format(entity, entity_type))
#                     entity_type = part_semantics.split("|")[1].split("=")[1]
#                     logger.info('{} {}'.format(entity, entity_type))
                    cluster_id = int(entity)
                    if part[-1] == ')':
#                         if (first_subtoken_idx, last_subtoken_idx) not in overlapping_span:
                        self.clusters[cluster_id].append((first_subtoken_idx, last_subtoken_idx, entity_type))
#                             overlapping_span[(first_subtoken_idx, last_subtoken_idx)] = True
                    else:
                        self.coref_stacks.append((cluster_id, first_subtoken_idx, entity_type))

#                     print(self.coref_stacks)

#                 else:
#                     parts = coref.split(")")
#                     for part in parts[1:]:
#                         cluster_id, start = self.coref_stacks.pop()
#                         print("Popped")
#                         print(self.coref_stacks)
#                         self.clusters[cluster_id].append((start, last_subtoken_idx))

            first_subtoken_idx += 1

        # Merge clusters if any clusters have common mentions
        merged_clusters = []
        for cluster in self.clusters.values():
            existing = None
            for mention in cluster:
                for merged_cluster in merged_clusters:
                    if mention in merged_cluster:
                        existing = merged_cluster
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often)")
                existing.update(cluster)
            else:
                merged_clusters.append(set(cluster))

        merged_clusters = [list(cluster) for cluster in merged_clusters]
        all_mentions = flatten(merged_clusters)
        sentence_map = get_sentence_map(self.segments, self.sentence_end)
        subtoken_map = flatten(self.segment_subtoken_map)

        # Sanity check
        assert len(all_mentions) == len(set(all_mentions))  # Each mention unique
        # Below should have length: # all subtokens with CLS, SEP in all segments
        num_all_seg_tokens = len(flatten(self.segments))
        assert num_all_seg_tokens == len(flatten(self.speakers))
        assert num_all_seg_tokens == len(subtoken_map)
        assert num_all_seg_tokens == len(sentence_map)

        return {
            "doc_key": self.doc_key,
            "tokens": self.tokens,
            "sentences": self.segments,
            "speakers": self.speakers,
            "constituents": [],
            "ner": [],
            "clusters": merged_clusters,
            'sentence_map': sentence_map,
            "subtoken_map": subtoken_map,
            'pronouns': self.pronouns
        }

def split_into_segments(document_state: DocumentState, max_seg_len, constraints1, constraints2, tokenizer):
    """ Split into segments.
        Add subtokens, subtoken_map, info for each segment; add CLS, SEP in the segment subtokens
        Input document_state: tokens, subtokens, token_end, sentence_end, utterance_end, subtoken_map, info
    """
    curr_idx = 0  # Index for subtokens
    prev_token_idx = 0
    while curr_idx < len(document_state.subtokens):
        # Try to split at a sentence end point
        end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)  # Inclusive
        while end_idx >= curr_idx and not constraints1[end_idx]:
            end_idx -= 1
        if end_idx < curr_idx:
            logger.info(f'{document_state.doc_key}: no sentence end found; split at token end')
            # If no sentence end point, try to split at token end point
            end_idx = min(curr_idx + max_seg_len - 1 - 2, len(document_state.subtokens) - 1)
            while end_idx >= curr_idx and not constraints2[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                logger.error('Cannot split valid segment: no sentence end or token end')

        segment = [tokenizer.cls_token] + document_state.subtokens[curr_idx: end_idx + 1] + [tokenizer.sep_token]
        document_state.segments.append(segment)

        subtoken_map = document_state.subtoken_map[curr_idx: end_idx + 1]
        document_state.segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])

        document_state.segment_info.append([None] + document_state.info[curr_idx: end_idx + 1] + [None])

        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[-1]


def get_doc_markables(doc_name, doc_lines, extract_MIN, keep_bridging, word_column=1, markable_column=10, bridging_column=11, print_debug=False):
    markables_cluster = {}
    markables_start = {}
    markables_end = {}
    markables_MIN = {}
    markables_coref_tag = {}
    markables_split = {} # set_id: [markable_id_1, markable_id_2 ...]
    bridging_antecedents = {}
    all_words = []
    stack = []
    for word_index, line in enumerate(doc_lines):
        columns = line.split()
        all_words.append(columns[word_column])

        if columns[markable_column] != '_':
            markable_annotations = columns[markable_column].split("(")
            semantics_annotations = columns[13].split("(") # modified
            if markable_annotations[0]:
                #the close bracket
                for _ in range(len(markable_annotations[0])):
                    markable_id = stack.pop()
                    markables_end[markable_id] = word_index

            for markable_annotation in markable_annotations[1:]:
                if markable_annotation.endswith(')'):
                    single_word = True
                    markable_annotation = markable_annotation[:-1]
                else:
                    single_word = False
                markable_info = {p[:p.find('=')]:p[p.find('=')+1:] for p in markable_annotation.split('|')}
                semantics_info = {p[:p.find('=')]:p[p.find('=')+1:] for p in semantics_annotations.split('|')}
                markable_id = markable_info['MarkableID']
                cluster_id = markable_info['EntityID']
                entity_type = semantics_info['Entity_Type']
                markables_cluster[markable_id] = cluster_id
                markables_start[markable_id] = word_index
                if single_word:
                    markables_end[markable_id] = word_index
                else:
                    stack.append(markable_id)

                markables_MIN[markable_id] = None
                if extract_MIN and 'Min' in markable_info:
                    MIN_Span = markable_info['Min'].split(',')
                    if len(MIN_Span) == 2:
                        MIN_start = int(MIN_Span[0]) - 1
                        MIN_end = int(MIN_Span[1]) - 1
                    else:
                        MIN_start = int(MIN_Span[0]) - 1
                        MIN_end = MIN_start
#                     markables_MIN[markable_id] = (MIN_start,MIN_end)
                    markables_MIN[markable_id] = (MIN_start,MIN_end,entity_type)

                markables_coref_tag[markable_id] = 'referring'
                if cluster_id.endswith('-Pseudo'):
                    markables_coref_tag[markable_id] = 'non_referring'

                if 'ElementOf' in markable_info:
                    element_of = markable_info['ElementOf'].split(',') # for markable participate in multiple plural using , split the element_of, e.g. ElementOf=1,2
                    for ele_of in element_of:
                        if ele_of not in markables_split:
                            markables_split[ele_of] = []
                        markables_split[ele_of].append(markable_id)
        if keep_bridging and columns[bridging_column] != '_':
            bridging_annotations = columns[bridging_column].split("(")
            for bridging_annotation in bridging_annotations[1:]:
                if bridging_annotation.endswith(')'):
                    bridging_annotation = bridging_annotation[:-1]
                bridging_info = {p[:p.find('=')]:p[p.find('=')+1:] for p in bridging_annotation.split('|')}
                bridging_antecedents[bridging_info['MarkableID']] = bridging_info['MentionAnchor']




    clusters = {}
    id2markable = {}
    for markable_id in markables_cluster:
        m = Markable(
                doc_name, markables_start[markable_id],
                markables_end[markable_id], markables_MIN[markable_id],
                markables_coref_tag[markable_id],
                all_words[markables_start[markable_id]:
                        markables_end[markable_id] + 1])
        id2markable[markable_id] = m
        if markables_cluster[markable_id] not in clusters:
            clusters[markables_cluster[markable_id]] = (
                    [], markables_coref_tag[markable_id],doc_name,[markables_cluster[mid] for mid in markables_split.get(markables_cluster[markable_id],[])])
        clusters[markables_cluster[markable_id]][0].append(m)

    bridging_pairs = {}
    for anaphora, antecedent in bridging_antecedents.items():
        if not anaphora in id2markable or not antecedent in id2markable:
            print('Skip bridging pair ({}, {}) as markable_id does not exist in identity column!'.format(antecedent,anaphora))
            continue
        bridging_pairs[id2markable[anaphora]] = id2markable[antecedent]

    #print([(str(ana),str(ant)) for ana,ant in bridging_pairs.items()])
    # for cid in clusters:
    #	 cl = clusters[cid]
    #	 print(cid,[str(m) for m in cl[0]],cl[1],cl[2],cl[3] )
    return clusters, bridging_pairs

def process_clusters(clusters, keep_singletons, keep_non_referring, keep_split_antecedent):
    removed_non_referring = 0
    removed_singletons = 0
    processed_clusters = []
    processed_non_referrings = []

    for cluster_id, (cluster, ref_tag, doc_name, split_cid_list) in clusters.items():
        #recusively find the split singular cluster
        if split_cid_list and keep_split_antecedent:
            # if using split-antecedent, we shouldn't remove singletons as they might be used by split-antecedents
            assert keep_singletons
            split_clusters = set()
            queue = deque()
            queue.append(cluster_id)
            while queue:
                curr = queue.popleft()
                curr_cl, curr_ref_tag, doc_name, curr_cid_list = clusters[curr]
                #non_referring shouldn't be used as split-antecedents
                # if curr_ref_tag != 'referring':
                #	 print(curr_ref_tag, doc_name, curr_cid_list)
                if curr_cid_list:
                    for c in curr_cid_list:
                        queue.append(c)
                else:
                    split_clusters.add(tuple(curr_cl))
            split_m = Markable(
                doc_name, -1,
                -1, None,
                'referring',
                '',
                is_split_antecedent=True,
                split_antecedent_members=split_clusters)

            cluster.append(split_m) #add the split_antecedents

        if ref_tag == 'non_referring':
            if keep_non_referring:
                processed_non_referrings.append(cluster[0])
            else:
                removed_non_referring += 1
            continue

        if not keep_singletons and len(cluster) == 1:
            removed_singletons += 1
            continue

        processed_clusters.append(cluster)

    if keep_split_antecedent:
        #step 2 merge equivalent split-antecedents clusters
        merged_clusters = []
        for cl in processed_clusters:
            existing = None
            for m in cl:
                if m.is_split_antecedent:
                #only do this for split-antecedents
                    for c2 in merged_clusters:
                        if m in c2:
                            existing = c2
                            break
            if existing:
                # print('merge cluster ', [str(m) for m in cl], ' and ', [str(m) for m in existing])
                existing.update(cl)
            else:
                merged_clusters.append(set(cl))
        merged_clusters = [list(cl) for cl in merged_clusters]
    else:
        merged_clusters = processed_clusters

    return (merged_clusters, processed_non_referrings,
            removed_non_referring, removed_singletons)

def get_all_docs(path):
    all_doc_sents = {}
    all_docs = {}
    doc_lines = []
    sentences = []
    sentence = []
    doc_name = None
    for line in open(path):
        line = line.strip()
        if line.startswith('# newdoc'):
            if doc_name and doc_lines:
                all_docs[doc_name] = doc_lines
                all_doc_sents[doc_name] = sentences
                doc_lines = []
                sentences = []
            cur_spk = "_"
            doc_name = line[len('# newdoc id = '):]
        elif "# speaker = " in line:
            cur_spk = line[len('# speaker = '):]
        elif line.startswith('#'):
            continue
        elif len(line) == 0:
            # MODIFIED
            doc_lines.append(line)
            # END
            sentences.append(sentence)
            sentence = []
            continue
        else:
            splt_line = line.split()
            splt_line[9] = "_".join(cur_spk.split())
            line = " ".join(splt_line)
            doc_lines.append(line)
            sentence.append(line.split()[1])

    sentences.append(sentence)
    if doc_name and doc_lines:
        all_docs[doc_name] = doc_lines
        all_doc_sents[doc_name] = sentences
    return all_docs, all_doc_sents

def get_markable_assignments(clusters):
    markable_cluster_ids = {}
    for cluster_id, cluster in enumerate(clusters):
        for m in cluster:
            markable_cluster_ids[m] = cluster_id
    return markable_cluster_ids

def get_document(doc_key, doc_lines, language, seg_len, tokenizer):
    """ Process raw input to finalized documents """
    document_state = UADocumentState(doc_key)
    word_idx = -1

    # Build up documents
    for line in doc_lines:
#         print(line)
        row = line.split()  # Columns for each token
        if len(row) == 0:
            document_state.sentence_end[-1] = True
        else:
            word_idx += 1
            word = normalize_word(row[1], language)
            subtokens = tokenizer.tokenize(word)
            document_state.tokens.append(word)
            document_state.token_end += [False] * (len(subtokens) - 1) + [True]
            for idx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                info = None if idx != 0 else (row + [len(subtokens)])
                document_state.info.append(info)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(word_idx)

    # Split documents
    constraits1 = document_state.sentence_end if language != 'arabic' else document_state.token_end
    split_into_segments(document_state, seg_len, constraits1, document_state.token_end, tokenizer)
    document = document_state.finalize()
    return document
