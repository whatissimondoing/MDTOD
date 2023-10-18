import os
import copy
import glob
import math
import time
import shutil
import fitlog
from abc import *
from collections import OrderedDict

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers.optimization import Adafactor
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, get_constant_schedule

from model import T5Base
from utils import definitions
from evaluator import MultiWozEvaluator, IncarEvaluator
from utils.io_utils import get_or_create_logger, save_json
from reader import MultiWOZReader, MultiWOZDataset, CollatorTrain

logger = get_or_create_logger(__name__)


class Reporter(object):
    def __init__(self, log_frequency, model_dir):
        self.log_frequency = log_frequency

        self.global_step = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0

        self.belief_loss = 0.0
        self.resp_loss = 0.0
        self.dual_belief_loss = 0.0
        self.dual_resp_loss = 0.0

        self.belief_correct = 0.0
        self.resp_correct = 0.0
        self.dual_belief_correct = 0.0
        self.dual_resp_correct = 0.0

        self.belief_count = 0.0
        self.resp_count = 0.0
        self.dual_belief_count = 0.0
        self.dual_resp_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        if "belief" in step_outputs:
            self.belief_loss += step_outputs["belief"]["loss"]
            self.belief_correct += step_outputs["belief"]["correct"]
            self.belief_count += step_outputs["belief"]["count"]
            do_belief_stats = True
        else:
            do_belief_stats = False

        if "resp" in step_outputs:
            self.resp_loss += step_outputs["resp"]["loss"]
            self.resp_correct += step_outputs["resp"]["correct"]
            self.resp_count += step_outputs["resp"]["count"]
            do_resp_stats = True
        else:
            do_resp_stats = False

        if "belief_to_utter" in step_outputs:
            self.dual_belief_loss += step_outputs["belief_to_utter"]["loss"]
            self.dual_belief_correct += step_outputs["belief_to_utter"]["correct"]
            self.dual_belief_count += step_outputs["belief_to_utter"]["count"]
            do_dual_belief = True
        else:
            do_dual_belief = False

        if "resp_to_utter" in step_outputs:
            self.dual_resp_loss += step_outputs["resp_to_utter"]["loss"]
            self.dual_resp_correct += step_outputs["resp_to_utter"]["correct"]
            self.dual_resp_count += step_outputs["resp_to_utter"]["count"]
            do_dual_resp = True
        else:
            do_dual_resp = False

        if is_train:
            self.lr = lr

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step, do_belief_stats, do_resp_stats, do_dual_belief, do_dual_resp)

    def info_stats(self, data_type, global_step, do_belief_stats=False, do_resp_stats=False, do_dual_belief=False, do_dual_resp=False):
        avg_step_time = self.step_time / self.log_frequency

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        if do_belief_stats:
            try:
                belief_ppl = math.exp(self.belief_loss / self.belief_count)
            except:
                belief_ppl = 0
                print("belief loss and belief count is {}, {}".format(self.belief_loss, self.belief_count))
            belief_acc = (self.belief_correct / self.belief_count) * 100

            if data_type == "train":
                fitlog.add_metric({"train": {"belief_acc": belief_acc}}, step=global_step)
            else:
                fitlog.add_metric({"dev": {"belief_acc": belief_acc}}, step=global_step)

            belief_info = "[belief] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(self.belief_loss, belief_ppl, belief_acc)
        else:
            belief_info = ""

        if do_dual_belief:
            dual_belief_ppl = math.exp(self.dual_belief_loss / self.dual_belief_count)
            dual_belief_acc = (self.dual_belief_correct / self.dual_belief_count) * 100
            if data_type == "train":
                fitlog.add_loss(self.dual_belief_loss, name="dual_belief_loss", step=global_step)
                fitlog.add_metric({"train": {"dual_belief_acc": dual_belief_acc}}, step=global_step)
            else:
                fitlog.add_metric({"dev": {"dual_belief_acc": dual_belief_acc}}, step=global_step)

            dual_belief_info = "[dual_belief] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(self.dual_belief_loss, dual_belief_ppl, dual_belief_acc)

        else:
            dual_belief_info = ""

        if do_dual_resp:
            dual_resp_ppl = math.exp(self.dual_resp_loss / self.dual_resp_count)
            dual_resp_acc = (self.dual_resp_correct / self.dual_resp_count) * 100
            if data_type == "train":
                fitlog.add_loss(self.dual_resp_loss, name="dual_resp_loss", step=global_step)
                fitlog.add_metric({"train": {"dual_resp_acc": dual_resp_acc}}, step=global_step)
            else:
                fitlog.add_metric({"dev": {"dual_resp_acc": dual_resp_acc}}, step=global_step)

            dual_resp_info = "[dual_resp] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(self.dual_resp_loss, dual_resp_ppl, dual_resp_acc)

        else:
            dual_resp_info = ""

        if do_resp_stats:
            resp_ppl = math.exp(self.resp_loss / self.resp_count)
            resp_acc = (self.resp_correct / self.resp_count) * 100
            if data_type == "train":
                fitlog.add_loss(self.resp_loss, name="loss", step=global_step)
                fitlog.add_metric({"train": {"resp_acc": resp_acc}}, step=global_step)
            else:
                fitlog.add_metric({"dev": {"resp_acc": resp_acc}}, step=global_step)

            resp_info = "[resp] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(self.resp_loss, resp_ppl, resp_acc)

        else:
            resp_info = ""

        logger.info(" ".join([common_info, belief_info, resp_info, dual_belief_info, dual_resp_info]))

        self.init_stats()


class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg, reader):
        self.cfg = cfg
        self.reader = reader

        self.pbar = None
        self.model = self.load_model()

    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
            initialize_additional_decoder = False
        elif self.cfg.train_from is not None:
            model_path = self.cfg.train_from
            initialize_additional_decoder = False
        else:
            model_path = self.cfg.backbone
            initialize_additional_decoder = True

        logger.info("Load models from {}".format(model_path))

        if self.cfg.dataset == "multiwoz":
            model_wrapper = T5Base  # TODO
            # model_wrapper = DialogLED
        else:
            model_wrapper = T5Base

        self.cfg.vocab_size = self.reader.vocab_size

        model = model_wrapper.from_pretrained(model_path, cfg=self.cfg)

        model.resize_token_embeddings(self.reader.vocab_size)

        if initialize_additional_decoder:
            model.initialize_additional_decoder()

        model.to(self.cfg.device)
        if self.cfg.num_gpus > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.cfg.local_rank], output_device=self.cfg.local_rank,
                                                        find_unused_parameters=False)

        return model

    def save_model(self, epoch):
        latest_ckpt = "ckpt-epoch{}".format(epoch)
        save_path = os.path.join(self.cfg.model_dir, latest_ckpt)

        models = self.model

        # model = self.model

        models.save_pretrained(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # keep chekpoint up to maximum
        checkpoints = sorted(
            glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
            key=os.path.getmtime,
            reverse=True)

        checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]

        for ckpt in checkpoints_to_be_deleted:
            shutil.rmtree(ckpt)

        return latest_ckpt

    def get_optimizer_and_scheduler(self, num_traininig_steps_per_epoch, train_batch_size):
        '''
        num_train_steps = (num_train_examples *
            self.cfg.epochs) // (train_batch_size * self.cfg.grad_accum_steps)
        '''
        num_train_steps = (num_traininig_steps_per_epoch * self.cfg.epochs) // self.cfg.grad_accum_steps

        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

        logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))

        self.pbar = tqdm(total=num_train_steps, desc="training")

        if self.cfg.optimizer == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        else:
            optimizer = Adafactor(self.model.parameters(),
                                  lr=1e-4,
                                  clip_threshold=1.0,
                                  warmup_init=False,
                                  scale_parameter=False,
                                  relative_step=False)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)
        # scheduler = None

        return optimizer, scheduler

    def count_tokens(self, pred, label, pad_id):
        label_mask = label.eq(pad_id).bool()
        pred[label_mask] = -1

        num_count = label.view(-1).ne(pad_id).long().sum()
        num_correct = torch.eq(pred.view(-1), label.view(-1)).long().sum()

        return num_correct, num_count

    def count_spans(self, pred, label):
        pred = pred.view(-1, 2)

        num_count = label.ne(-1).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    def count_modes(self, pred, label):
        # pred = pred.view(-1, 2)

        num_count = label.shape[0]
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, global_step):
        raise NotImplementedError


class MultiWOZRunner(BaseRunner):
    def __init__(self, cfg):
        reader = MultiWOZReader(cfg.backbone, cfg.version, cfg.para_num, cfg.dataset)

        super(MultiWOZRunner, self).__init__(cfg, reader)

        self.kl_scheduler = None

    # @profile
    def step_fn(self, global_step, inputs, belief_labels, resp_labels):
        attention_mask = torch.where(inputs.eq(self.reader.pad_token_id), 0, 1)

        belief_outputs = self.model(input_ids=inputs,
                                    attention_mask=attention_mask,
                                    lm_labels=belief_labels,
                                    return_dict=False,
                                    decoder_type="belief")

        belief_loss = belief_outputs[0]
        belief_pred = belief_outputs[1]

        dst_to_utter_loss, resp_to_utter_loss = 0, 0
        dst_to_utter_correct, dst_to_utter_count = 0, 0
        resp_to_utter_correct, resp_to_utter_count = 0, 0

        if self.cfg.add_du_dual:
            belief_attention_mask = torch.where(belief_labels.eq(self.reader.pad_token_id), 0, 1)
            utterance_outputs = self.model(input_ids=belief_labels,
                                           attention_mask=belief_attention_mask,
                                           lm_labels=inputs,
                                           return_dict=False,
                                           decoder_type="belief")
            dst_to_utter_loss = utterance_outputs[0]
            dst_to_utter_pred = utterance_outputs[1]

            dst_to_utter_correct, dst_to_utter_count = self.count_tokens(
                dst_to_utter_pred, inputs, pad_id=self.reader.pad_token_id)

        resp_loss = 0
        if self.cfg.task == "e2e":
            last_hidden_state = belief_outputs[2]

            encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_state)
            resp_outputs = self.model(encoder_outputs=encoder_outputs,
                                      lm_labels=resp_labels,
                                      return_dict=False,
                                      decoder_type="resp"
                                      )

            resp_loss = resp_outputs[0]
            resp_pred = resp_outputs[1]

            num_resp_correct, num_resp_count = self.count_tokens(
                resp_pred, resp_labels, pad_id=self.reader.pad_token_id)

            if self.cfg.add_ru_dual:
                resp_attention_mask = torch.where(resp_labels.eq(self.reader.pad_token_id), 0, 1)
                resp_to_utter_outputs = self.model(input_ids=resp_labels,
                                                   attention_mask=resp_attention_mask,
                                                   lm_labels=inputs,
                                                   return_dict=False,
                                                   decoder_type="resp")
                resp_to_utter_loss = resp_to_utter_outputs[0]
                resp_to_utter_pred = resp_to_utter_outputs[1]

                resp_to_utter_correct, resp_to_utter_count = self.count_tokens(
                    resp_to_utter_pred, inputs, pad_id=self.reader.pad_token_id)

        else:
            num_resp_correct, num_resp_count = 0, 0

        num_belief_correct, num_belief_count = self.count_tokens(
            belief_pred, belief_labels, pad_id=self.reader.pad_token_id)

        loss = belief_loss

        if self.cfg.task == "e2e" and self.cfg.resp_loss_coeff > 0:
            loss += (self.cfg.resp_loss_coeff * resp_loss)  # TODO

        if self.cfg.add_du_dual:
            loss += (self.cfg.du_coeff * dst_to_utter_loss)

        if self.cfg.add_ru_dual:
            loss += (self.cfg.ru_coeff * resp_to_utter_loss)

        step_outputs = {"belief": {"loss": belief_loss.item(),
                                   "correct": num_belief_correct.item(),
                                   "count": num_belief_count.item()}}

        if self.cfg.add_du_dual:
            step_outputs["belief_to_utter"] = {"loss": dst_to_utter_loss.item(),
                                               "correct": dst_to_utter_correct.item(),
                                               "count": dst_to_utter_count.item()}

        if self.cfg.task == "e2e":
            step_outputs["resp"] = {"loss": resp_loss.item(),
                                    "correct": num_resp_correct.item(),
                                    "count": num_resp_count.item()}

            if self.cfg.add_ru_dual:
                step_outputs["resp_to_utter"] = {"loss": resp_to_utter_loss.item(),
                                                 "correct": resp_to_utter_correct.item(),
                                                 "count": resp_to_utter_count.item()}

        return loss, step_outputs

    def train_epoch(self, train_iterator, optimizer, scheduler, reporter=None):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        global_step = reporter.global_step if reporter else 0

        for step, batch in enumerate(train_iterator):
            start_time = time.time()

            inputs, belief_labels, resp_labels = batch
            inputs = inputs.to(self.cfg.device)
            belief_labels = belief_labels.to(self.cfg.device)
            resp_labels = resp_labels.to(self.cfg.device)

            loss, step_outputs = self.step_fn(global_step, inputs, belief_labels, resp_labels)

            if self.cfg.grad_accum_steps > 1:
                loss = loss / self.cfg.grad_accum_steps

            loss.backward()

            if self.cfg.optimizer != "adafactor":
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                self.pbar.update(1)

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]

                if reporter is not None:
                    reporter.step(start_time, lr, step_outputs)

    def train(self):
        num_workers = 0
        train_dataset = MultiWOZDataset(self.cfg, self.reader, "train", self.cfg.task, self.cfg.ururu, context_size=self.cfg.context_size,
                                        num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains,
                                        train_ratio=self.cfg.train_ratio, para_num=self.cfg.para_num)
        train_sampler = DistributedSampler(train_dataset) if self.cfg.num_gpus > 1 else RandomSampler(train_dataset)
        collator = CollatorTrain(self.reader.pad_token_id, self.reader.tokenizer, self.cfg)
        self.train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.cfg.batch_size, collate_fn=collator,
                                           num_workers=4 * self.cfg.num_gpus, pin_memory=True)

        dev_dataset = MultiWOZDataset(self.cfg, self.reader, "dev", self.cfg.task, self.cfg.ururu, context_size=self.cfg.context_size,
                                      num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains)
        dev_sampler = DistributedSampler(dev_dataset) if self.cfg.num_gpus > 1 else RandomSampler(dev_dataset)
        self.dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=3 * self.cfg.batch_size, collate_fn=collator,
                                         num_workers=num_workers)

        num_training_steps_per_epoch = len(self.train_dataloader)  # len_dataloader = len_datasets // (num_gpus * batch_size)

        optimizer, scheduler = self.get_optimizer_and_scheduler(num_training_steps_per_epoch, self.cfg.batch_size)

        if self.cfg.local_rank in [0, -1]:
            reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir)
        else:
            reporter = None

        current_min_dev = 1e8
        best_combine_score = 0.0

        # scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.use_fp16)

        for epoch in range(1, self.cfg.epochs + 1):

            if self.cfg.num_gpus > 1:
                self.train_dataloader.sampler.set_epoch(epoch)

            self.train_epoch(self.train_dataloader, optimizer, scheduler, reporter)

            if self.cfg.local_rank in [0, -1]:
                logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))
                self.save_model(epoch)

                if not self.cfg.no_validation:
                    self.cfg.ckpt = os.path.join(self.cfg.model_dir, "ckpt-epoch" + str(epoch))
                    if self.cfg.train_ratio == 1 and self.cfg.task == "e2e":
                        results = self.predict(reporter.global_step)
                        metric_name = "jga_test" if "jga" in results else "test"
                        if results["score"] > best_combine_score:
                            best_combine_score = results["score"]
                            fitlog.add_best_metric({metric_name: results})
                    else:
                        dev_loss = self.validation(reporter.global_step)
                        if dev_loss < current_min_dev:
                            current_min_dev = dev_loss
                            results = self.predict(reporter.global_step)
                            metric_name = "jga_test" if "jga" in results else "test"
                            fitlog.add_best_metric({metric_name: results})

        self.pbar.close()

    def validation(self, global_step):
        self.model.eval()

        reporter = Reporter(1000000, self.cfg.model_dir)
        with torch.no_grad():
            for batch in tqdm(self.dev_dataloader, total=len(self.dev_dataloader), desc="Validation"):
                start_time = time.time()

                inputs, belief_labels, resp_labels = batch
                inputs = inputs.to(self.cfg.device)
                belief_labels = belief_labels.to(self.cfg.device)
                resp_labels = resp_labels.to(self.cfg.device)

                loss, step_outputs = self.step_fn(global_step, inputs, belief_labels, resp_labels)

                reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

            do_belief_stats = True if "belief" in step_outputs else False
            do_resp_stats = True if "resp" in step_outputs else False
            do_dual_belief = True if "belief_to_utter" in step_outputs else False
            do_dual_response = True if "resp_to_utter" in step_outputs else False

            dev_loss = reporter.resp_loss if do_resp_stats else reporter.belief_loss
            reporter.info_stats("dev", global_step, do_belief_stats, do_resp_stats, do_dual_belief, do_dual_response)

        return dev_loss

    def finalize_bspn(self, belief_outputs, domain_history, constraint_history, input_ids=None):
        eos_token_id = self.reader.get_token_id(definitions.EOS_BELIEF_TOKEN)

        batch_decoded = []
        for i, belief_output in enumerate(belief_outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_output = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            bspn = belief_output[:eos_idx + 1]

            decoded = {}

            decoded["bspn_gen"] = bspn

            batch_decoded.append(decoded)

        return batch_decoded

    def finalize_resp(self, resp_outputs):
        bos_action_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
        eos_action_token_id = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)

        bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
        eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

        batch_decoded = []
        for resp_output in resp_outputs:
            resp_output = resp_output[1:]
            if self.reader.eos_token_id in resp_output:
                eos_idx = resp_output.index(self.reader.eos_token_id)
                resp_output = resp_output[:eos_idx]

            try:
                bos_action_idx = resp_output.index(bos_action_token_id)
                eos_action_idx = resp_output.index(eos_action_token_id)
            except ValueError:
                # logger.warn("bos/eos action token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                aspn = [bos_action_token_id, eos_action_token_id]
            else:
                aspn = resp_output[bos_action_idx:eos_action_idx + 1]

            try:
                bos_resp_idx = resp_output.index(bos_resp_token_id)
                eos_resp_idx = resp_output.index(eos_resp_token_id)
            except ValueError:
                # logger.warn("bos/eos resp token not in : {}".format(
                #     self.reader.tokenizer.decode(resp_output)))
                resp = [bos_resp_token_id, eos_resp_token_id]
            else:
                resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

            decoded = {"aspn_gen": aspn, "resp_gen": resp}

            batch_decoded.append(decoded)

        return batch_decoded

    def predict(self, global_step):
        self.model.eval()

        if self.cfg.num_gpus > 1:
            model = self.model.module
        else:
            model = self.model

        test_dataset = MultiWOZDataset(self.cfg, self.reader, "test", self.cfg.task, self.cfg.ururu, context_size=self.cfg.context_size,
                                       num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains)
        pred_batches = test_dataset.all_batches

        early_stopping = True if self.cfg.beam_size > 1 else False

        eval_dial_list = None
        if self.cfg.excluded_domains is not None:
            eval_dial_list = []

            for domains, dial_ids in test_dataset.dial_by_domain.items():
                domain_list = domains.split("-")

                # if len(set(domain_list) & set(self.cfg.excluded_domains)) == 0:
                #     eval_dial_list.extend(dial_ids)
                if len(set(domain_list) & set(self.cfg.excluded_domains)) > 0:  # Zero-shot prediction for excluded domains
                    eval_dial_list.extend(dial_ids)  # 里面放的是除外的domain的数据

        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)
            dial_history = [[] for _ in range(batch_size)]
            domain_history = [[] for _ in range(batch_size)]
            constraint_dicts = [OrderedDict() for _ in range(batch_size)]
            for turn_batch in test_dataset.transpose_batch(dial_batch):
                batch_encoder_input_ids = []
                for t, turn in enumerate(turn_batch):
                    if self.reader.dataset == "incar":
                        turn["resp"] = turn["redx"]
                    context, _ = test_dataset.flatten_dial_history(
                        dial_history[t], [], len(turn["user"]), self.cfg.context_size)

                    encoder_input_ids = context + turn["user"] + [self.reader.eos_token_id]

                    batch_encoder_input_ids.append(test_dataset.tensorize(encoder_input_ids))

                    turn_domain = turn["turn_domain"][-1]

                    if "[" in turn_domain:
                        turn_domain = turn_domain[1:-1]

                    domain_history[t].append(turn_domain)

                batch_encoder_input_ids = pad_sequence(batch_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)

                # batch_encoder_input_ids = batch_encoder_input_ids.to(self.cfg.device)

                attention_mask = torch.where(batch_encoder_input_ids.eq(self.reader.pad_token_id), 0, 1)

                # belief tracking
                with torch.no_grad():
                    encoder_outputs = model(input_ids=batch_encoder_input_ids,
                                            attention_mask=attention_mask,
                                            return_dict=False,
                                            encoder_only=True)

                    encoder_hidden_states = encoder_outputs

                    if isinstance(encoder_hidden_states, tuple):
                        last_hidden_state = encoder_hidden_states[0]
                    else:
                        last_hidden_state = encoder_hidden_states

                    # wrap up encoder outputs
                    encoder_outputs = BaseModelOutput(
                        last_hidden_state=last_hidden_state)

                    belief_outputs = model.generate(encoder_outputs=encoder_outputs,
                                                    attention_mask=attention_mask,
                                                    eos_token_id=self.reader.eos_token_id,
                                                    max_length=100,
                                                    do_sample=self.cfg.do_sample,
                                                    num_beams=self.cfg.beam_size,
                                                    early_stopping=early_stopping,
                                                    temperature=self.cfg.temperature,
                                                    top_k=self.cfg.top_k,
                                                    top_p=self.cfg.top_p,
                                                    decoder_type="belief")

                belief_outputs = belief_outputs.cpu().numpy().tolist()

                input_ids = None

                decoded_belief_outputs = self.finalize_bspn(belief_outputs, domain_history, constraint_dicts, input_ids)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_belief_outputs[t])

                if self.cfg.task == "e2e":
                    dbpn = []

                    if self.cfg.use_true_dbpn:
                        for turn in turn_batch:
                            dbpn.append(turn["dbpn"])
                    else:
                        for turn in turn_batch:
                            bspn_gen = turn["bspn_gen"]

                            bspn_gen = self.reader.tokenizer.decode(
                                bspn_gen, clean_up_tokenization_spaces=False)

                            db_token = self.reader.bspn_to_db_pointer(bspn_gen,
                                                                      turn["turn_domain"])

                            dbpn_gen = self.reader.encode_text(
                                db_token,
                                bos_token=definitions.BOS_DB_TOKEN,
                                eos_token=definitions.EOS_DB_TOKEN)

                            turn["dbpn_gen"] = dbpn_gen

                            dbpn.append(dbpn_gen)

                    for t, db in enumerate(dbpn):
                        if self.cfg.use_true_curr_aspn:
                            db += turn_batch[t]["aspn"]

                        # T5 use pad_token as start_decoder_token_id
                        dbpn[t] = [self.reader.pad_token_id] + db

                    # print(dbpn)

                    # aspn has different length
                    if self.cfg.use_true_curr_aspn:
                        for t, _dbpn in enumerate(dbpn):
                            resp_decoder_input_ids = test_dataset.tensorize([_dbpn])

                            resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                            encoder_outputs = BaseModelOutput(
                                last_hidden_state=last_hidden_state[t].unsqueeze(0))

                            with torch.no_grad():
                                resp_outputs = model.generate(
                                    encoder_outputs=encoder_outputs,
                                    attention_mask=attention_mask[t].unsqueeze(0),
                                    decoder_input_ids=resp_decoder_input_ids,
                                    eos_token_id=self.reader.eos_token_id,
                                    max_length=200,
                                    do_sample=self.cfg.do_sample,
                                    num_beams=self.cfg.beam_size,
                                    early_stopping=early_stopping,
                                    temperature=self.cfg.temperature,
                                    top_k=self.cfg.top_k,
                                    top_p=self.cfg.top_p,
                                    decoder_type="resp")

                                resp_outputs = resp_outputs.cpu().numpy().tolist()

                                decoded_resp_outputs = self.finalize_resp(resp_outputs)

                                turn_batch[t].update(**decoded_resp_outputs[0])

                    else:
                        resp_decoder_input_ids = test_dataset.tensorize(dbpn)

                        resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)

                        # response generation
                        with torch.no_grad():
                            resp_outputs = model.generate(
                                encoder_outputs=encoder_outputs,
                                attention_mask=attention_mask,
                                decoder_input_ids=resp_decoder_input_ids,
                                eos_token_id=self.reader.eos_token_id,
                                max_length=200,
                                do_sample=self.cfg.do_sample,
                                num_beams=self.cfg.beam_size,
                                early_stopping=early_stopping,
                                temperature=self.cfg.temperature,
                                top_k=self.cfg.top_k,
                                top_p=self.cfg.top_p,
                                decoder_type="resp")

                        resp_outputs = resp_outputs.cpu().numpy().tolist()

                        decoded_resp_outputs = self.finalize_resp(resp_outputs)

                        for t, turn in enumerate(turn_batch):
                            turn.update(**decoded_resp_outputs[t])

                # update dial_history
                for t, turn in enumerate(turn_batch):
                    pv_text = copy.copy(turn["user"])

                    if self.cfg.use_true_prev_bspn:
                        pv_bspn = turn["bspn"]
                    else:
                        pv_bspn = turn["bspn_gen"]

                    if self.cfg.use_true_dbpn:
                        pv_dbpn = turn["dbpn"]
                    else:
                        pv_dbpn = turn["dbpn_gen"]

                    if self.cfg.use_true_prev_aspn:
                        pv_aspn = turn["aspn"]
                    else:
                        pv_aspn = turn["aspn_gen"]

                    if self.cfg.use_true_prev_resp:
                        if self.cfg.task == "e2e":
                            pv_resp = turn["redx"]
                        else:
                            pv_resp = turn["resp"]
                    else:
                        pv_resp = turn["resp_gen"]

                    if self.cfg.ururu:
                        pv_text += pv_resp
                    else:
                        pv_text += (pv_bspn + pv_dbpn + pv_aspn + pv_resp)

                    dial_history[t].append(pv_text)

            result = test_dataset.get_readable_batch(dial_batch)
            results.update(**result)

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        if self.reader.dataset == "incar":
            evaluator = IncarEvaluator(self.reader, self.cfg.pred_data_type)
        else:
            evaluator = MultiWozEvaluator(self.reader, self.cfg.pred_data_type)

        if self.cfg.task == "e2e":
            bleu, success, match = evaluator.e2e_eval(results, eval_dial_list=eval_dial_list)

            score = 0.5 * (success + match) + bleu

            logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (match, success, bleu, score))
            fitlog.add_metric({"test": {"inform": match, "success": success, "bleu": bleu, "score": score}}, step=global_step)

            return {"inform": match, "success": success, "bleu": bleu, "score": score}
        else:
            joint_goal, f1, accuracy, count_dict, correct_dict = evaluator.dialog_state_tracking_eval(results)

            fitlog.add_metric({"test": {"jga": joint_goal, "f1": f1, "acc": accuracy}}, step=global_step)

            logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;' % (
                joint_goal, accuracy, f1))

            for domain_slot, count in count_dict.items():
                correct = correct_dict.get(domain_slot, 0)

                acc = (correct / count) * 100

                logger.info('{0} acc: {1:.2f}'.format(domain_slot, acc))

            return {"jga": joint_goal, "f1": f1, "acc": accuracy}
