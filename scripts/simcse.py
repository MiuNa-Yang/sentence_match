from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from conf.base_conf import TrainerConf, DataModuleConf
from conf.model_conf import ModelConf
from dataset import data_augment
from dataset.session_dataset import SessionDataset
from dataset.sts_data import STSDataset
from model import Tokenizer
from model.base_model import BaseModel, Similarity
from utils import metrics, schedulers
from utils.json_config import json_config


@json_config
@dataclass
class SimCSETaskConfig(ModelConf, TrainerConf, DataModuleConf):
    val_data_path: str = ''
    mlm_flag: bool = True
    mlm_weight: float = 0.1
    temp: float = 0.05


class SimCSE(LightningModule):
    def __init__(self, config: SimCSETaskConfig, from_pretrained=False):
        super().__init__()
        self.save_hyperparameters(ignore='pretrained')
        self.config = config
        self.model = BaseModel(config.model_name, from_pretrained=from_pretrained)
        if self.config.mlm_flag:
            self.mlm = BertLMPredictionHead(self.model.backbone.config)
        self.sim = Similarity(self.config.temp)
        self.loss = CrossEntropyLoss()

        self.tokenizer = Tokenizer(config.tokenizer_name, model_max_length=self.config.model_max_length)
        self.train_data = None
        self.val_data = None

    def forward(self, sent, mlm_label):
        outputs, pooler_output = self.model(**sent)
        pooler_output = pooler_output.view(pooler_output.size(0) // 2, 2, -1)
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]

        # sim loss
        if dist.is_initialized():
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        sim = self.sim(z1, z2)
        batch_size = sim.size(0)
        sim_label = torch.arange(batch_size, dtype=torch.long, device=self.device)
        sim_loss = self.loss(sim, sim_label)

        # mlm_head loss
        if self.config.mlm_flag:
            m1 = self.mlm(outputs)
            m1 = torch.permute(m1, [0, 2, 1])
            mlm_loss = self.loss(m1, mlm_label)
        else:
            mlm_loss = 0

        loss = sim_loss + self.config.mlm_weight * mlm_loss
        return sim, sim_loss, mlm_loss, loss

    def pair_forward(self, s1, s2, mlm_label_1=None, mlm_label_2=None, label=None):
        outputs_1, pooler_output_1 = self.model(**s1)
        outputs_2, pooler_output_2 = self.model(**s2)

        # sim loss
        sim = F.cosine_similarity(pooler_output_1, pooler_output_2)
        sim_loss = F.mse_loss(sim, label)

        # mlm loss
        if self.config.mlm_flag:
            m1 = self.mlm(outputs_1)
            m2 = self.mlm(outputs_2)
            m1 = torch.permute(m1, [0, 2, 1])
            m2 = torch.permute(m2, [0, 2, 1])
            mlm_loss = self.mlm_loss(m1, mlm_label_1) + self.mlm_loss(m2, mlm_label_2)
        else:
            mlm_loss = 0

        loss = sim_loss + self.config.mlm_weight * mlm_loss
        return sim, sim_loss, mlm_loss, loss

    def training_step(self, batch, batch_idx):
        sim, sim_loss, mlm_loss, loss = self(**batch)
        # print(sim_loss, mlm_loss, loss)
        self.log('train/sim_loss', sim_loss)
        self.log('train/mlm_loss', mlm_loss)
        self.log('train/loss', loss)
        self.log('lr', self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config.val_data_path:
            loss = self.supervise_validation_step(batch, batch_idx)
        else:
            loss = self.unsupervised_validation_step(batch, batch_idx)
        return loss

    def unsupervised_validation_step(self, batch, batch_idx):
        s1, s2, mlm_label_1, mlm_label_2 = batch['s1'], batch['s2'], batch['mlm_label_1'], batch['mlm_label_2']
        sim, sim_loss, mlm_loss, loss = self(s1, s2, mlm_label_1, mlm_label_2)
        self.log('val/sim_loss', sim_loss)
        self.log('val/mlm_loss', mlm_loss)
        self.log('val/loss', loss)

        batch_size = sim.size(0)
        labels = torch.arange(batch_size, dtype=torch.long)

        log_dict = {}
        log_dict.update(metrics.top_k_accuracy_score(sim, labels, 4))
        log_dict.update(metrics.top_k_accuracy_score(sim, labels, 16))

        similarity_flatten = sim.view(-1).tolist()
        label_flatten = torch.eye(batch_size).view(-1).tolist()
        log_dict.update(metrics.spearman_corr(similarity_flatten, label_flatten))
        self.log_dict(log_dict)
        return loss

    def supervise_validation_step(self, batch, batch_idx):
        s1, s2, mlm_label_1, mlm_label_2, label = \
            batch['s1'], batch['s2'], batch['mlm_label_1'], batch['mlm_label_2'], batch['label']
        sim, sim_loss, mlm_loss, loss = self.pair_forward(s1, s2, mlm_label_1, mlm_label_2, label)
        self.log('val/sim_loss', sim_loss)
        self.log('val/mlm_loss', mlm_loss)
        self.log('val/loss', loss)

        log_dict = {}
        log_dict.update(metrics.spearman_corr(sim, label))
        self.log_dict(log_dict)
        return loss

    def configure_optimizers(self):
        optimizer, scheduler = schedulers.get_scheduler(self.model, self.config)
        return [optimizer], [scheduler]

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit':
            self.train_data = SessionDataset(self.config.data_path)
            print(f'train data: {len(self.train_data)}')
        if stage in ['fit', 'validate']:
            self.val_data = STSDataset(self.config.val_data_path)
            print(f'valid data: {len(self.val_data)}')

    def collate_fn(self, batch):
        s1, s2 = zip(*batch)
        s = list(s1) + list(2)
        s = self.tokenizer(s)
        if self.config.mlm_flag:
            s, mlm_label = data_augment.mlm_aug(s, drop_ratio=0.3)
        else:
            mlm_label = None
        return {'sent': s1, 'mlm_label': mlm_label}

    def collate_fn_val(self, batch):
        s1 = [_['s1'] for _ in batch]
        s2 = [_['s2'] for _ in batch]
        label = [_['label'] for _ in batch]
        s1 = self.tokenizer(s1)
        s2 = self.tokenizer(s2)
        label = torch.Tensor(label)
        if self.config.mlm_flag:
            s1, mlm_label_1 = data_augment.mlm_aug(s1, drop_ratio=0.3)
            s2, mlm_label_2 = data_augment.mlm_aug(s2, drop_ratio=0.3)
        else:
            mlm_label_1 = None
            mlm_label_2 = None
        return {'s1': s1, 's2': s2, 'mlm_label_1': mlm_label_1, 'mlm_label_2': mlm_label_2, 'label': label}

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config.train_batch_size,
                          shuffle=True, num_workers=self.config.num_workers,
                          collate_fn=self.collate_fn,
                          pin_memory=False, drop_last=True, persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.config.val_batch_size,
                          shuffle=True, num_workers=self.config.num_workers,
                          collate_fn=self.collate_fn_val,
                          pin_memory=True, drop_last=False, persistent_workers=True)
