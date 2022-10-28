from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer


class Tokenizer:
    def __init__(self, path, return_tensors='pt', model_max_length=512):
        self.return_tensors = return_tensors
        self.t = BertTokenizer.from_pretrained(path)
        self.t.model_max_length = model_max_length

    def __call__(self, x):
        return self.t(x, return_tensors=self.return_tensors, padding='max_length', truncation=True)


if __name__ == '__main__':
    t = Tokenizer('../resources/pretrained_model/rbt3', model_max_length=20)
    s = t(['测试' * 20])
    print(s)
