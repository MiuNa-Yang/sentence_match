import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from torch import Tensor


class BaseModel(nn.Module):
    def __init__(self, model_name, from_pretrained=True):
        super().__init__()
        if from_pretrained:
            self.backbone = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        else:
            config = BertConfig.from_pretrained(model_name)
            self.backbone = BertModel(config, add_pooling_layer=False)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.backbone(input_ids, attention_mask, token_type_ids)[0]
        pooler_output = outputs[:, 0, :]
        return outputs, pooler_output


class Pooler(nn.Module):
    def __init__(self, hidden_size, output_size=None):
        super().__init__()
        if not output_size:
            output_size = hidden_size
        self.dense = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ModelwithPooler(BaseModel):
    def __init__(self, model_name, from_pretrained=True):
        super(ModelwithPooler, self).__init__(model_name, from_pretrained)
        self.hidden_size = self.backbone.config.hidden_size
        self.pooler = Pooler(self.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids, attention_mask, token_type_ids)[0]
        pooler_output = self.pooler(outputs)
        return pooler_output


class ClsModel(nn.Module):
    def __init__(self, model_name, class_num=2, from_pretrained=False):
        super().__init__()
        self.backbone = ModelwithPooler(model_name, from_pretrained=from_pretrained)
        self.hidden_size = self.backbone.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(self.hidden_size, class_num)

    def forward(self, **x):
        sent_emb = self.backbone(**x)
        z = self.dropout(sent_emb)
        outputs = self.cls(z)
        return outputs

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
