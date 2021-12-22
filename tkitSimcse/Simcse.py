# -*- coding: utf-8 -*-

"""
sim.py
用于计算两个序列的相关度
参考Sentence-BERT计算句子相似度的孪生网络


"""
import pytorch_lightning as pl
import torch
import torch.optim as optim
from tkitLr import CyclicCosineDecayLR
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, BertTokenizer


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits


class SimCSE(pl.LightningModule):
    """
    用于处理两个序列相关度的模型

    可以用于做数据相关或者下一句等任务

    参考simCse模式，加入双向gru来降低对参数的依赖。


    """

    def __init__(
            self, learning_rate=3e-4,
            T_max=5,
            ignore_index=0,
            optimizer_name="AdamW",
            dropout=0.2,
            labels=2,
            pretrained="uer/chinese_roberta_L-2_H-128",
            batch_size=2,
            trainfile="./data/train.pkt",
            valfile="./data/val.pkt",
            testfile="./data/test.pkt",
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        config = AutoConfig.from_pretrained(pretrained)
        self.model = AutoModel.from_pretrained(pretrained, config=config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        分类解决方案


        """
        B, L = input_ids.size()
        SL = 1
        NL = L - 1
        #         print(input_ids_a.size())
        # sim data
        outputs_a = self.model(input_ids=input_ids[:1], token_type_ids=token_type_ids[:1],
                               attention_mask=attention_mask[:1])
        emb_a = self.mean_pooling(outputs_a[0], attention_mask[:1])
        outputs_b = self.model(input_ids=input_ids[:1], token_type_ids=token_type_ids[:1],
                               attention_mask=attention_mask[:1])
        emb_b = self.mean_pooling(outputs_b[0], attention_mask[:1])


        # no sim
        outputs_c = self.model(input_ids=input_ids[1:], token_type_ids=token_type_ids[1:],
                               attention_mask=attention_mask[1:])
        emb_c = self.mean_pooling(outputs_c[0], attention_mask[1:])

        B_c, _ = attention_mask[1:].size()

        outputs_d = emb_a.repeat(B_c, 1)

        x = torch.cat((emb_a, outputs_d), 0)
        y = torch.cat((emb_b, emb_c), 0)

        cos = nn.CosineSimilarity(dim=-1, eps=1e-10)
        # cos_sim = 1 - cos(x, y)

        cos_sim = cos(x, y)
        labels = torch.Tensor([1] + [0] * B_c).to(self.device)
        loss = self.loss_fc(cos_sim, labels)
        print(cos_sim,labels)
        return loss

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        # print("token_embeddings",token_embeddings)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    #     def mean_pooling(self,model_output, attention_mask):
    #         token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    #         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    #         return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    #     def loss_fc(self,out,label):
    #         loss_fc = nn.L1Loss()
    #         loss=loss_fc(out,label)
    #         return loss
    def loss_fc(self, out, label):
        """
        jisuan 离散的损失

        """
        #         loss_fc = nn.CrossEntropyLoss()
        loss_fc = nn.BCEWithLogitsLoss()
        #         print(out.size(),label.size())
        loss = loss_fc(out.view(-1), label)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_ids, token_type_ids, attention_mask = batch

        # input_ids_a = self.tomask(input_ids_a)[0]
        # input_ids_b = self.tomask(input_ids_b)[0]
        loss = self(input_ids, token_type_ids, attention_mask)
        # loss = self.loss_fc(out, labels)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_ids, token_type_ids, attention_mask = batch

        # input_ids_a = self.tomask(input_ids_a)[0]
        # input_ids_b = self.tomask(input_ids_b)[0]
        loss = self(input_ids, token_type_ids, attention_mask)
        # loss = self.loss_fc(out, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_ids, token_type_ids, attention_mask = batch

        # input_ids_a = self.tomask(input_ids_a)[0]
        # input_ids_b = self.tomask(input_ids_b)[0]
        loss = self(input_ids, token_type_ids, attention_mask)
        # loss = self.loss_fc(out, labels)
        self.log('test_loss', loss)
        return loss

    def train_dataloader(self):
        train = torch.load(self.hparams.trainfile)
        return DataLoader(train, batch_size=int(self.hparams.batch_size), num_workers=2, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        val = torch.load(self.hparams.valfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=2, pin_memory=True)

    def test_dataloader(self):
        val = torch.load(self.hparams.testfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=2, pin_memory=True)

    def configure_optimizers(self):
        """优化器 自动优化器"""
        optimizer = getattr(optim, self.hparams.optimizer_name)(self.parameters(), lr=self.hparams.learning_rate)
        #         使用自适应调整模型
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5000, factor=0.8,
        #                                                        verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000,
        #                                                                  T_mult=2, eta_min=0, verbose=False)
        #         scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 2000, 500)
        #         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-8)
        #         scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=1e-3, T_max=1000, eta_min=1e-8)

        scheduler = CyclicCosineDecayLR(optimizer,
                                        init_decay_epochs=1000,
                                        min_decay_lr=1e-8,
                                        restart_interval=1000,
                                        restart_lr=self.hparams.learning_rate / 1,
                                        restart_interval_multiplier=1.5,
                                        warmup_epochs=1000,
                                        warmup_start_lr=self.hparams.learning_rate / 10)
        #
        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'name': "lr_scheduler",
            'monitor': 'train_loss',  # 监听数据变化
            'strict': True,
        }
        #         return [optimizer], [lr_scheduler]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


if __name__ == "__main__":
    # main()?
    print("模型初始化测试")

    model = SentenceModel(pretrained='/data/PycharmProjects/Sentence-BERT/data/model/')
    tokenizer = BertTokenizer.from_pretrained('/data/PycharmProjects/Sentence-BERT/data/model/')

    # Sentences we want sentence embeddings for
    sentences = ['天气真好', '天气不好']
    sentences_b = ['天气', '不好']
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding="max_length", max_length=64, truncation=True, return_tensors='pt')
    encoded_input_b = tokenizer(sentences_b, padding="max_length", max_length=64, truncation=True, return_tensors='pt')

    # labels测试
    labels = torch.Tensor([0, 1])
    output = model(encoded_input['input_ids'], encoded_input_b['input_ids'], encoded_input['attention_mask'],
                   encoded_input_b['attention_mask'])
    # 计算loss
    loss = model.loss_fc(output, labels)

    print(output, loss)
