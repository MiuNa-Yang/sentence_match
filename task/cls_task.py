import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from model.tokenizer import Tokenizer
from model.cls_model import ClsModel
from model.loss import FocalLoss
from dataset.cls_data import ClsDataset
from utils import metrics
from typing import List, Any, Optional
from model import schedulers
from utils.data_utils import random_split_by_ratio


class ClsTask(LightningModule):
    def __init__(self, cfg, from_pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = Tokenizer(path=cfg.tokenizer_name)
        self.model = ClsModel(self.cfg.model_name, self.cfg.clss_num, from_pretrained=from_pretrained)
        self.loss = FocalLoss()

        self.train_data = None
        self.val_data = None

    def forward(self, x, label):
        outputs = self.model(**x)
        loss = self.loss(outputs, label)

        return outputs, loss

    def training_step(self, batch, batch_idx):
        sent, label = batch['sent'], batch['label']
        outputs, loss = self(sent, label)
        self.log('train/loss', loss)
        self.log('lr', self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        sent, label = batch['sent'], batch['label']
        outputs, loss = self(sent, label)
        self.log('val/loss', loss)

        preds = torch.argmax(outputs, dim=1).tolist()
        label = label.tolist()

        metric_dict = metrics.cls_micro_metrics(preds, label)
        self.log_dict(metric_dict)
        return loss

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        preds = []
        for _preds in results[0]:
            preds += _preds
        results[0] = preds

    def configure_optimizers(self):
        optimizer, scheduler = schedulers.get_scheduler(self.model, self.cfg)
        return [optimizer], [scheduler]

    def setup(self, stage: Optional[str] = None):
        if stage in ['fit', 'validation']:
            data = ClsDataset(self.cfg.data_path)
            train, val = random_split_by_ratio(data, self.cfg.test_size)
            self.train_data = train
            self.val_data = val
            print(f'train data size: {len(self.train_data)}')
            print(f'val data size: {len(self.val_data)}')

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.cfg.train_batch_size,
                          shuffle=True, num_workers=self.cfg.num_workers,
                          collate_fn=self.collate_fn,
                          pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.cfg.val_batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers,
                          collate_fn=self.collate_fn,
                          pin_memory=True, drop_last=False, persistent_workers=True)

    def check_dataloader(self):
        return DataLoader(self.train_data, batch_size=2, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.cfg.predict_batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers,
                          collate_fn=self.collate_fn,
                          drop_last=False)

    def get_dataloader(self, data):
        return DataLoader(data, batch_size=self.cfg.predict_batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers,
                          collate_fn=self.collate_fn_pred)

    def collate_fn(self, batch):
        text = [_['sent'] for _ in batch]
        label = [_['label'] for _ in batch]

        text = self.tokenizer(text)
        label = torch.LongTensor(label)
        return {'text': text, 'label': label}

    def collate_fn_pred(self, batch):
        text = batch
        text = self.tokenizer(text)
        return {'text': text}
