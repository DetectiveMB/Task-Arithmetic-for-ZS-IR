# llama_ir

from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          PreTrainedModel,
                          LlamaTokenizer)
from typing import List, Union, Tuple, Mapping, Optional
from dataclasses import dataclass
from tqdm.autonotebook import trange
import torch
import copy

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

class LLamaRank:
    def __init__(self, 
                 model_path: str,
                 tokenizer = None,
                 use_amp = True,
                 token_false = None,
                 token_true  = None, 
                 device = None):
        self.model = self.get_model(model_path, device = device)
        self.tokenizer = LlamaTokenizer.from_pretrained('zyznull/RankingGPT-llama2-7b',trust_remote_code=True) # tokenizer or self.get_tokenizer(model_path)
        #self.token_false_id, self.token_true_id = self.get_prediction_tokens(
        #        model_path, self.tokenizer, token_false, token_true)
        self.model_path = model_path
        self.device = device #next(self.model.parameters(), None).device
        #self.use_amp = use_amp

    @staticmethod
    def get_model(model_path: str, *args, device: str = None, **kwargs) -> AutoModelForCausalLM:
        device = device #device or ('cuda:1' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return AutoModelForCausalLM.from_pretrained(model_path, *args, **kwargs).to(device).eval()
    
    @staticmethod
    def truncation(self,text,length):
        text=self.tokenizer.decode(self.tokenizer.encode(text,max_length=length, add_special_tokens=False))
        return text

    def _tokenize_fn(self,strings):
        tokenized_list = [
        self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
        )['input_ids']
        for text in strings
    ]
        input_ids = labels = [tokenized[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
        tokenized.ne(self.tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
        return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

    def predict(self, sentences: List[Tuple[str,str]], batch_size: int = 32, **kwargs) -> List[float]:
        
        sentence_dict, queries, scores = {}, [], []
       
        # T5 model requires a batch of single query and top-k documents
        scores_list = []
        for (query, doc_text) in sentences:
            if query not in sentence_dict:
                sentence_dict[query] = []
                queries.append(query) # Preserves order of queries
            sentence_dict[query].append(doc_text) 
        
        for start_idx in trange(0, len(queries), 1): # Take one query at a time
            batch_input = (queries[start_idx], sentence_dict[queries[start_idx]]) # (single query, top-k docs) 
            all_examples=[]
            all_sources=[]
            all_queries=[]
            for document in batch_input[1]:
                prompt='Document: {doc} Query:'
                try:
                    source = prompt.format(doc = self.truncation(self, document, 256)).replace(DEFAULT_PAD_TOKEN,'PAD')
                except:
                    import ipdb
                    ipdb.set_trace()
                query = self.truncation(self, batch_input[0], 128)
                all_examples.append(source+query)
                all_sources.append(source)
                all_queries.append(query)

            for index in range(0,len(all_examples),25):
                examples=all_examples[index:index+25]
                sources=all_sources[index:index+25]
                #queries_bis=all_queries[index:index+25]
                examples_tokenized, sources_tokenized = [self._tokenize_fn(strings) for strings in (examples, sources)]
                input_ids = examples_tokenized["input_ids"]
        
                labels = copy.deepcopy(input_ids)
            
                for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
                    label[:source_len] = IGNORE_INDEX
        

                for index in range(len(input_ids)):
                    input_ids[index]=input_ids[index][:-1]

                input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).cuda()
        
                labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX).cuda()
                labels = labels[..., 1:].contiguous() #BL

                with torch.no_grad():
                    lm_logits = self.model(input_ids=input_ids,attention_mask=input_ids.ne(self.tokenizer.pad_token_id))[0]
                    preds = torch.nn.functional.log_softmax(lm_logits,dim=-1)
                    label_no_ingore = torch.where(labels==-100,torch.ones(labels.shape).long().cuda(),labels)
                    logprobs = torch.gather(preds, -1, label_no_ingore.unsqueeze(dim=-1)).squeeze(dim=-1) # B L
                    indexs=(labels!=-100).long()
                    scores=(logprobs*indexs).sum(dim=-1)/indexs.sum(dim=-1)
                    scores=scores.cpu().tolist()
                    scores_list+=scores
        
        assert len(scores_list) == len(sentences) # Sanity check, should be equal
        return scores_list