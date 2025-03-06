from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
from mmengine.model import BaseModule
from embodiedqa.registry import MODELS
'''
'roberta-base'
'bert-base-uncased'
'sentence-transformers/all-mpnet-base-v2'
'''
@MODELS.register_module()
class TextModelWrapper(BaseModule):
    def __init__(self, name="roberta-base", frozen=True,learnable_parameter_keys=[]):
        """
        Initialize a TextModelWrapper.

        Args:
            name (str): The name of the transformer model to use.
            frozen (bool): Whether to freeze the text model parameters.
            learnable_parameter_keys (list[str]): The keys of the parameters
                that are set to be learnable.
        """
        super().__init__()
        self.config = AutoConfig.from_pretrained(name)
        # 判断模型类型
        if  name in ['facebook/bart-base']:
            # 加载BART模型的Encoder部分
            self.text_model = AutoModel.from_pretrained(name, config=self.config).get_encoder()
        else:
            self.text_model = AutoModel.from_pretrained(name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.frozen = frozen
        if frozen:
            for name, param in self.text_model.named_parameters():
                if not any(k in name for k in learnable_parameter_keys):
                    param.requires_grad = False

    def forward(self, *args, **kwargs):
        return self.text_model(*args, **kwargs)

    def get_tokenizer(self):
        return self.tokenizer
    def get_word_embeddings(self):
        return self.text_model.embeddings.word_embeddings
