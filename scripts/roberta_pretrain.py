import torch
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from conf.simcse_conf import TaskConf
from model.base_model import BaseModel
from utils.schedulers import get_scheduler


class RobertaPretrainTask(LightningModule):
    def __init__(self, conf: TaskConf, from_pretrained):
        super().__init__()
        self.conf = conf
        self.model = BaseModel(conf.model_name, from_pretrained=from_pretrained)
        self.mlm_head = BertLMPredictionHead(self.model.backbone.config)
        self.loss = CrossEntropyLoss()

    def forward(self, sent, label):
        outputs = self.model(sent)[0]
        x = self.mlm_head(outputs)
        m1 = torch.permute(x, [0, 2, 1])
        mlm_loss = self.loss(m1, label)
        return mlm_loss

    def training_step(self, batch, batch_idx):
        loss = self(**batch)
        self.log('train/loss', loss)
        self.log('lr', self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(**batch)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer, scheduler = get_scheduler(self.model, self.conf)
        return [optimizer], [scheduler]
