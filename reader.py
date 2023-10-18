import os
import math
import spacy
import random
import difflib
from tqdm import tqdm
from difflib import get_close_matches
from itertools import chain
from collections import OrderedDict, defaultdict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer, BartTokenizer

from utils import definitions_mwz as definitions
# from utils import definitions

from utils.io_utils import load_json, load_pickle, save_pickle, get_or_create_logger, save_json
from external_knowledges import MultiWozDB

logger = get_or_create_logger(__name__)


class BaseReader(object):
    def __init__(self, backbone):
        self.nlp = spacy.load("en_core_web_sm")

        self.tokenizer = self.init_tokenizer(backbone)

        self.data_dir = self.get_data_dir()

        encoded_data_path = os.path.join(self.data_dir, "encoded_data.pkl")

        if os.path.exists(encoded_data_path):
            # if 1 == 0:
            logger.info("Load encoded data from {}".format(encoded_data_path))

            self.data = load_pickle(encoded_data_path)

        else:
            logger.info("Encoding data and save to {}".format(encoded_data_path))
            train = self.encode_data("train")
            dev = self.encode_data("dev")
            test = self.encode_data("test")

            self.data = {"train": train, "dev": dev, "test": test}

            save_pickle(self.data, encoded_data_path)

        span_tokens = [self.pad_token, "O"]
        for slot in definitions.EXTRACTIVE_SLOT:
            span_tokens.append(slot)

        self.span_tokens = span_tokens

    def get_data_dir(self):
        raise NotImplementedError

    def init_tokenizer(self, backbone):
        if "t5" in backbone.lower():
            tokenizer = T5Tokenizer.from_pretrained(backbone)
        else:
            tokenizer = BartTokenizer.from_pretrained(backbone)

        special_tokens = []

        # add domains
        domains = definitions.ALL_DOMAINS + ["general"]
        for domain in sorted(domains):
            token = "[" + domain + "]"
            special_tokens.append(token)

        # add intents
        intents = list(set(chain(*definitions.DIALOG_ACTS.values())))
        for intent in sorted(intents):
            token = "[" + intent + "]"
            special_tokens.append(token)

        # add slots
        slots = list(set(definitions.ALL_INFSLOT + definitions.ALL_REQSLOT))

        for slot in sorted(slots):
            token = "[value_" + slot + "]"
            special_tokens.append(token)

        special_tokens.extend(definitions.SPECIAL_TOKENS)

        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        return tokenizer

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    @property
    def max_seq_len(self):
        return self.tokenizer.model_max_length

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def encode_text(self, text, bos_token=None, eos_token=None):
        tokens = text.split() if isinstance(text, str) else text

        assert isinstance(tokens, list)

        if bos_token is not None:
            if isinstance(bos_token, str):
                bos_token = [bos_token]

            tokens = bos_token + tokens

        if eos_token is not None:
            if isinstance(eos_token, str):
                eos_token = [eos_token]

            tokens = tokens + eos_token

        encoded_text = self.tokenizer.encode(" ".join(tokens))

        # except eos token
        if encoded_text[-1] == self.eos_token_id:
            encoded_text = encoded_text[:-1]

        return encoded_text

    def encode_data(self, data_type):
        raise NotImplementedError


class MultiWOZReader(BaseReader):
    def __init__(self, backbone, version, para_num, dataset):
        self.version = version
        self.dataset = dataset

        self.db = MultiWozDB(os.path.join(os.path.dirname(self.get_data_dir()), "db"), dataset)
        self.para_num = para_num

        super(MultiWOZReader, self).__init__(backbone)

    def get_data_dir(self):
        if self.dataset == "incar":
            return os.path.join(
                "data", "Kvret", "processed_1"
            )
        else:
            return os.path.join(
                "data", "MultiWOZ_{}".format(self.version), "processed")

    def encode_data(self, data_type):
        data = load_json(os.path.join(self.data_dir, "{}_data.json".format(data_type)))

        encoded_data = []
        for fn, dial in tqdm(data.items(), desc=data_type, total=len(data)):

            # if "_" in fn and int(fn[0]) >= self.para_num:  # Control the number of paraphrase-augmented data
            #     continue

            encoded_dial = []

            accum_constraint_dict = {}
            for t in dial["log"]:
                turn_constrain_dict = self.bspn_to_constraint_dict(t["constraint"])
                for domain, sv_dict in turn_constrain_dict.items():
                    if domain not in accum_constraint_dict:
                        accum_constraint_dict[domain] = {}

                    for s, v in sv_dict.items():
                        if s not in accum_constraint_dict[domain]:
                            accum_constraint_dict[domain][s] = []

                        accum_constraint_dict[domain][s].append(v)

            prev_bspn = ""
            for idx, t in enumerate(dial["log"]):
                enc = {}
                enc["dial_id"] = fn
                enc["turn_num"] = t["turn_num"]
                enc["turn_domain"] = t["turn_domain"].split()
                enc["pointer"] = [int(i) for i in t["pointer"].split(",")]

                target_domain = enc["turn_domain"][0] if len(enc["turn_domain"]) == 1 else enc["turn_domain"][1]

                target_domain = target_domain[1:-1]

                user_ids = self.encode_text(t["user"],
                                            bos_token=definitions.BOS_USER_TOKEN,
                                            eos_token=definitions.EOS_USER_TOKEN)

                enc["user"] = user_ids

                usdx_ids = self.encode_text(t["user_delex"],
                                            bos_token=definitions.BOS_USER_TOKEN,
                                            eos_token=definitions.EOS_USER_TOKEN)
                # enc["usdx"] = usdx_ids  # TODO

                resp_ids = self.encode_text(t["nodelx_resp"],
                                            bos_token=definitions.BOS_RESP_TOKEN,
                                            eos_token=definitions.EOS_RESP_TOKEN)

                enc["resp"] = resp_ids

                redx_ids = self.encode_text(t["resp"],
                                            bos_token=definitions.BOS_RESP_TOKEN,
                                            eos_token=definitions.EOS_RESP_TOKEN)

                enc["redx"] = redx_ids

                '''
                bspn_ids = self.encode_text(t["constraint"],
                                            bos_token=definitions.BOS_BELIEF_TOKEN,
                                            eos_token=definitions.EOS_BELIEF_TOKEN)

                enc["bspn"] = bspn_ids
                '''

                constraint_dict = self.bspn_to_constraint_dict(t["constraint"])
                ordered_constraint_dict = OrderedDict()
                for domain, slots in definitions.INFORMABLE_SLOTS.items():
                    if domain not in constraint_dict:
                        continue

                    ordered_constraint_dict[domain] = OrderedDict()
                    for slot in slots:
                        if slot not in constraint_dict[domain]:
                            continue

                        value = constraint_dict[domain][slot]

                        ordered_constraint_dict[domain][slot] = value

                ordered_bspn = self.constraint_dict_to_bspn(ordered_constraint_dict)

                bspn_ids = self.encode_text(ordered_bspn,
                                            bos_token=definitions.BOS_BELIEF_TOKEN,
                                            eos_token=definitions.EOS_BELIEF_TOKEN)

                enc["bspn"] = bspn_ids

                aspn_ids = self.encode_text(t["sys_act"],
                                            bos_token=definitions.BOS_ACTION_TOKEN,
                                            eos_token=definitions.EOS_ACTION_TOKEN)

                enc["aspn"] = aspn_ids

                pointer = enc["pointer"][:-2]
                if not any(pointer):
                    db_token = definitions.DB_NULL_TOKEN
                else:
                    db_token = "[db_{}]".format(pointer.index(1))

                dbpn_ids = self.encode_text(db_token,
                                            bos_token=definitions.BOS_DB_TOKEN,
                                            eos_token=definitions.EOS_DB_TOKEN)

                enc["dbpn"] = dbpn_ids

                if (len(enc["user"]) == 0 or len(enc["resp"]) == 0 or
                        len(enc["redx"]) == 0 or len(enc["bspn"]) == 0 or
                        len(enc["aspn"]) == 0 or len(enc["dbpn"]) == 0):
                    raise ValueError(fn, idx)

                # NOTE: if curr_constraint_dict does not include span[domain][slot], remove span[domain][slot] ??

                user_span = self.get_span(
                    target_domain,
                    self.tokenizer.convert_ids_to_tokens(user_ids),
                    self.tokenizer.convert_ids_to_tokens(usdx_ids),
                    accum_constraint_dict)

                enc["user_span"] = user_span

                resp_span = self.get_span(
                    target_domain,
                    self.tokenizer.convert_ids_to_tokens(resp_ids),
                    self.tokenizer.convert_ids_to_tokens(redx_ids),
                    accum_constraint_dict)

                enc["resp_span"] = resp_span

                encoded_dial.append(enc)

                prev_bspn = t["constraint"]

                # prev_constraint_dict = curr_constraint_dict

            encoded_data.append(encoded_dial)

        return encoded_data

    def bspn_to_constraint_dict(self, bspn):
        bspn = bspn.split() if isinstance(bspn, str) else bspn

        constraint_dict = OrderedDict()
        domain, slot = None, None
        for token in bspn:
            if token == definitions.EOS_BELIEF_TOKEN:
                break

            if token.startswith("["):
                token = token[1:-1]

                if token in definitions.ALL_DOMAINS:
                    domain = token

                if token.startswith("value_"):
                    if domain is None:
                        continue

                    if domain not in constraint_dict:
                        constraint_dict[domain] = OrderedDict()

                    slot = token.split("_")[1]

                    constraint_dict[domain][slot] = []

            else:
                try:
                    if domain is not None and slot is not None:
                        constraint_dict[domain][slot].append(token)
                except KeyError:
                    continue

        for domain, sv_dict in constraint_dict.items():
            for s, value_tokens in sv_dict.items():
                constraint_dict[domain][s] = " ".join(value_tokens)

        return constraint_dict

    def constraint_dict_to_bspn(self, constraint_dict):
        tokens = []
        for domain, sv_dict in constraint_dict.items():
            tokens.append("[" + domain + "]")
            for s, v in sv_dict.items():
                tokens.append("[value_" + s + "]")
                tokens.extend(v.split())

        return " ".join(tokens)

    def get_span(self, domain, text, delex_text, constraint_dict):
        span_info = {}

        if domain not in constraint_dict:
            return span_info

        tokens = text.split() if isinstance(text, str) else text

        delex_tokens = delex_text.split() if isinstance(delex_text, str) else delex_text

        seq_matcher = difflib.SequenceMatcher()

        seq_matcher.set_seqs(tokens, delex_tokens)

        for opcode in seq_matcher.get_opcodes():
            tag, i1, i2, j1, j2 = opcode

            lex_tokens = tokens[i1: i2]
            delex_token = delex_tokens[j1: j2]

            if tag == "equal" or len(delex_token) != 1:
                continue

            delex_token = delex_token[0]

            if not delex_token.startswith("[value_"):
                continue

            slot = delex_token[1:-1].split("_")[1]

            if slot not in definitions.EXTRACTIVE_SLOT:
                continue

            value = self.tokenizer.convert_tokens_to_string(lex_tokens)

            if slot in constraint_dict[domain] and value in constraint_dict[domain][slot]:
                if domain not in span_info:
                    span_info[domain] = {}

                span_info[domain][slot] = (i1, i2)

        return span_info

    def bspn_to_db_pointer(self, bspn, turn_domain):
        constraint_dict = self.bspn_to_constraint_dict(bspn)

        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith("[") else match_dom
        match = matnums[match_dom]

        vector = self.db.addDBIndicator(match_dom, match)

        return vector

    def canonicalize_span_value(self, domain, slot, value, cutoff=0.6):
        ontology = self.db.extractive_ontology

        if domain not in ontology or slot not in ontology[domain]:
            return value

        candidates = ontology[domain][slot]

        matches = get_close_matches(value, candidates, n=1, cutoff=cutoff)

        if len(matches) == 0:
            return value
        else:
            return matches[0]


class CollatorTrain(object):
    def __init__(self, pad_token_id, tokenizer, cfg):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, batch):
        batch_encoder_input_ids = []
        batch_belief_label_ids = []
        batch_resp_label_ids = []
        batch_size = len(batch)

        for i in range(batch_size):
            encoder_input_ids, belief_label_ids, resp_label_ids = batch[i]

            batch_encoder_input_ids.append(torch.tensor(encoder_input_ids, dtype=torch.long))
            batch_belief_label_ids.append(torch.tensor(belief_label_ids, dtype=torch.long))
            batch_resp_label_ids.append(torch.tensor(resp_label_ids, dtype=torch.long))

        batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids, batch_first=True, padding_value=self.pad_token_id)
        batch_belief_label_ids = pad_sequence(batch_belief_label_ids, batch_first=True, padding_value=self.pad_token_id)
        batch_resp_label_ids = pad_sequence(batch_resp_label_ids, batch_first=True, padding_value=self.pad_token_id)

        return batch_encoder_input_ids, batch_belief_label_ids, batch_resp_label_ids


class MultiWOZDataset(Dataset):
    def __init__(self, cfg, reader, data_type, task, ururu, context_size=-1, num_dialogs=-1, excluded_domains=None, train_ratio=1.0, para_num=-1):
        super().__init__()
        self.reader = reader
        self.task = task
        self.ururu = ururu
        self.context_size = context_size
        self.dials = self.reader.data[data_type]
        if self.reader.dataset == "multiwoz":
            self.dial_by_domain = load_json("data/MultiWOZ_2.1/dial_by_domain.json")
        else:
            self.dial_by_domain = self.count_domain()
        self.cfg = cfg

        if excluded_domains is not None:
            if data_type in ["train", "dev"]:
                logger.info("Exclude domains in {}: {}".format(data_type, excluded_domains))
            else:
                logger.info("Domains for predictions in {}: {}".format(data_type, excluded_domains))

            target_dial_ids = []
            for domains, dial_ids in self.dial_by_domain.items():
                domain_list = domains.split("-")

                if len(set(domain_list) & set(excluded_domains)) == 0:
                    target_dial_ids.extend(dial_ids)

            temp_dial = []
            if data_type in ["train", "dev"]:
                for d in self.dials:
                    temp_dial_id = d[0]["dial_id"] if "_" not in d[0]["dial_id"] else d[0]["dial_id"][2:]
                    if temp_dial_id in target_dial_ids:
                        temp_dial.append(d)
                self.dials = temp_dial
                # self.dials = [d for d in self.dials if d[0]["dial_id"] in target_dial_ids]
            else:  # Zero-shot Prediction for excluded domains
                for d in self.dials:
                    temp_dial_id = d[0]["dial_id"] if "_" not in d[0]["dial_id"] else d[0]["dial_id"][2:]
                    if temp_dial_id not in target_dial_ids:
                        temp_dial.append(d)
                self.dials = temp_dial
                # self.dials = [d for d in self.dials if d[0]["dial_id"] not in target_dial_ids]

        if data_type == "train":
            if num_dialogs > 0:
                self.dials = random.sample(self.dials, min(num_dialogs, len(self.dials)))

            if num_dialogs == -1:
                train_num = math.ceil(len(self.dials) * train_ratio)
                self.dials = random.sample(self.dials, min(train_num, len(self.dials)))

        self.para_num = para_num

        if data_type in ["train", "dev"]:
            self.create_turn_batch()
        else:
            self.create_pred_batch(self.cfg.batch_size)

    def tensorize(self, ids):
        return torch.tensor(ids, dtype=torch.long, device=self.cfg.device)

    def count_domain(self):
        dial_by_domains = {}
        for dial in self.dials:
            domain = dial[0]["turn_domain"][0]
            if '[' in domain:
                domain = domain[1:-1]

            if domain not in dial_by_domains:
                dial_by_domains[domain] = []
            dial_id = dial[0]["dial_id"] if "_" not in dial[0]["dial_id"] else dial[0]["dial_id"][2:]
            if dial_id not in dial_by_domains[domain]:
                dial_by_domains[domain].append(dial_id)
        return dial_by_domains

    def create_turn_batch(self):
        logger.info("Creating turn batches...")
        self.turn_encoder_input_ids = []
        self.turn_belief_label_ids = []
        self.turn_resp_label_ids = []

        for dial in tqdm(self.dials, desc='Creating turn batches'):

            if "_" in dial[0]['dial_id'] and int(dial[0]['dial_id'][0]) >= self.para_num:  # Control the number of paraphrase-augmented data
                continue

            dial_history = []
            span_history = []

            for turn in dial:
                context, span_dict = self.flatten_dial_history(dial_history, span_history, len(turn['user']), self.context_size)
                encoder_input_ids = context + turn['user'] + [self.reader.eos_token_id]

                bspn = turn['bspn']
                bspn_label = bspn
                belief_label_ids = bspn_label + [self.reader.eos_token_id]

                resp = turn['dbpn'] + turn['aspn'] + turn['redx']
                resp_label_ids = resp + [self.reader.eos_token_id]

                self.turn_encoder_input_ids.append(encoder_input_ids)
                self.turn_belief_label_ids.append(belief_label_ids)
                self.turn_resp_label_ids.append(resp_label_ids)

                turn_span_info = {}
                for domain, ss_dict in turn['user_span'].items():
                    for s, span in ss_dict.items():
                        if domain not in turn_span_info:
                            turn_span_info[domain] = {}

                        if s not in turn_span_info[domain]:
                            turn_span_info[domain][s] = []

                        turn_span_info[domain][s].append(span)

                if self.task == 'dst':
                    for domain, ss_dict in turn['resp_span'].items():
                        for s, span in ss_dict.items():
                            if domain not in turn_span_info:
                                turn_span_info[domain] = {}

                            if s not in turn_span_info[domain]:
                                turn_span_info[domain][s] = []

                            adjustment = len(turn["user"])

                            if not self.ururu:
                                adjustment += (len(bspn) + len(turn['dbpn']) + len(turn['aspn']))

                            start_idx = span[0] + adjustment
                            end_idx = span[1] + adjustment

                            turn_span_info[domain][s].append((start_idx, end_idx))

                if self.ururu:
                    if self.task == 'dst':
                        turn_text = turn['user'] + turn['resp']
                    else:
                        turn_text = turn['user'] + turn['redx']
                else:
                    if self.task == 'dst':
                        turn_text = turn['user'] + bspn + turn['dbpn'] + turn['aspn'] + turn['resp']
                    else:
                        turn_text = turn['user'] + bspn + turn['dbpn'] + turn['aspn'] + turn['redx']

                dial_history.append(turn_text)
                span_history.append(turn_span_info)

    def create_pred_batch(self, batch_size):
        def bucket_by_turn(encoded_data):
            turn_bucket = {}
            for dial in encoded_data:
                turn_len = len(dial)
                if turn_len not in turn_bucket:
                    turn_bucket[turn_len] = []

                turn_bucket[turn_len].append(dial)

            return OrderedDict(sorted(turn_bucket.items(), key=lambda i: i[0]))

        def construct_mini_batch(data, batch_size, num_gpus):
            all_batches = []
            batch = []
            for dial in data:
                batch.append(dial)
                if len(batch) == batch_size:
                    all_batches.append(batch)
                    batch = []

            # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
            if (len(batch) % num_gpus) != 0:
                batch = batch[:-(len(batch) % num_gpus)]
            if len(batch) > 0.5 * batch_size:
                all_batches.append(batch)
            elif len(all_batches):
                all_batches[-1].extend(batch)
            else:
                all_batches.append(batch)
            return all_batches

        turn_bucket = bucket_by_turn(self.dials)  # 按照session中turn_num进行排序
        self.all_batches = []
        for k in turn_bucket:
            # if data_type != "test" and (k == 1 or k >= 17):
            #     continue
            if k == 1:
                continue

            batches = construct_mini_batch(turn_bucket[k], batch_size, 1)  # TODO change num_gpus
            self.all_batches += batches

        logger.info("batches for predictions : {}".format(len(self.all_batches)))

    def transpose_batch(self, dial_batch):
        turn_batch = []
        turn_num = len(dial_batch[0])
        for turn in range(turn_num):
            turn_l = []
            for dial in dial_batch:
                this_turn = dial[turn]
                turn_l.append(this_turn)
            turn_batch.append(turn_l)
        return turn_batch

    def flatten_dial_history(self, dial_history, span_history, len_postfix, context_size):
        if context_size > 0:
            context_size -= 1

        if context_size == 0:
            windowed_context = []
            windowed_span_history = []
        elif context_size > 0:
            windowed_context = dial_history[-context_size:]
            windowed_span_history = span_history[-context_size:]
        else:
            windowed_context = dial_history
            windowed_span_history = span_history

        ctx_len = sum([len(c) for c in windowed_context])

        # consider eos_token
        spare_len = self.reader.max_seq_len - len_postfix - 1
        while 0 <= spare_len <= ctx_len:
            ctx_len -= len(windowed_context[0])
            windowed_context.pop(0)
            if len(windowed_span_history) > 0:
                windowed_span_history.pop(0)

        context_span_info = defaultdict(list)
        for t, turn_span_info in enumerate(windowed_span_history):
            for domain, span_info in turn_span_info.items():
                if isinstance(span_info, dict):
                    for slot, spans in span_info.items():
                        adjustment = 0

                        if t > 0:
                            adjustment += sum([len(c)
                                               for c in windowed_context[:t]])

                        for span in spans:
                            start_idx = span[0] + adjustment
                            end_idx = span[1] + adjustment

                            context_span_info[slot].append((start_idx, end_idx))

                elif isinstance(span_info, list):
                    slot = domain
                    spans = span_info

                    adjustment = 0
                    if t > 0:
                        adjustment += sum([len(c)
                                           for c in windowed_context[:t]])

                    for span in spans:
                        start_idx = span[0] + adjustment
                        end_idx = span[1] + adjustment

                        context_span_info[slot].append((start_idx, end_idx))

        context = list(chain(*windowed_context))

        return context, context_span_info

    def get_readable_batch(self, dial_batch):
        dialogs = {}

        decoded_keys = ["user", "resp", "redx", "bspn", "aspn", "dbpn",
                        "bspn_gen", "bspn_gen_with_span",
                        "dbpn_gen", "aspn_gen", "resp_gen"]
        for dial in dial_batch:
            dial_id = dial[0]["dial_id"]

            dialogs[dial_id] = []

            for turn in dial:
                readable_turn = {}

                for k, v in turn.items():
                    if k == "dial_id":
                        continue
                    elif k in decoded_keys:
                        v = self.reader.tokenizer.decode(
                            v, clean_up_tokenization_spaces=False)
                        '''
                        if k == "user":
                            print(k, v)
                        '''
                    elif k == "pointer":
                        turn_doamin = turn["turn_domain"][-1]
                        v = self.reader.db.pointerBack(v, turn_doamin)
                    if k == "user_span" or k == "resp_span":
                        speaker = k.split("_")[0]
                        v_dict = {}
                        for domain, ss_dict in v.items():
                            v_dict[domain] = {}
                            for s, span in ss_dict.items():
                                v_dict[domain][s] = self.reader.tokenizer.decode(
                                    turn[speaker][span[0]: span[1]])
                        v = v_dict

                    readable_turn[k] = v

                dialogs[dial_id].append(readable_turn)

        return dialogs

    def __len__(self):
        return len(self.turn_encoder_input_ids)

    def __getitem__(self, index):
        return self.turn_encoder_input_ids[index], self.turn_belief_label_ids[index], self.turn_resp_label_ids[index]


if __name__ == '__main__':
    pass
