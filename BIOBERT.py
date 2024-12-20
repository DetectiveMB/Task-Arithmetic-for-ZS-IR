# BIOBERT

from beir import util, LoggingHandler
from beir.retrieval import models
from sentence_transformers import models as sentence_models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
import logging
import pathlib, os
from transformers import BertTokenizer, BertModel
from fire import Fire
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "nfcorpus"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")



#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#bert = BertModel.from_pretrained("bert-base-cased")


#biobert = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

#mmsmarco = BertModel.from_pretrained('castorini/mdpr-tied-pft-msmarco')

#msmarco = BertModel.from_pretrained('Capreolus/bert-base-msmarco')

import torch, os


def extract(
    base_model_path: str,
    chat_model_path: str,
    output_path: str,
    l1: float,
):
    base_model = AutoModel.from_pretrained(base_model_path, torch_dtype='auto')
    ft_model = AutoModel.from_pretrained(chat_model_path, torch_dtype='auto')
    
    chat_vector_params = {
        #'chat_embed': base_model.get_input_embeddings().weight,
        #'chat_lmhead': base_model.get_output_embeddings().weight,
        'chat_vector': {},
        'cfg': {
            'base_model_path': base_model_path,
            'chat_model_path': chat_model_path,
        }
    }
    
    for (n1, p1), (n2, p2) in zip(base_model.named_parameters(),ft_model.named_parameters()):

        chat_vector_params['chat_vector'][n1] =  p2 - l1*p1
        #print(n1==n2)
        
    
    os.makedirs(output_path, exist_ok=True)
    torch.save(
        chat_vector_params,
        f"{output_path}/pytorch_model.bin",
    )

#sentence-transformers/msmarco-bert-base-dot-v5 #NeuML/pubmedbert-base-embeddings




def add_chat_vector(
    base: PreTrainedModel,
    chat_vector_path: str,
    ratio: float,
    #skip_embed: bool = False,
    #special_tokens_map: dict[int, int] = None
):
    chat_vector = torch.load(f'{chat_vector_path}/pytorch_model.bin')
    
    new_state_dict = {}
    
    #print(special_tokens_map)
    for n, para in base.named_parameters():
        #para = para + ratio * chat_vector['chat_vector'][n.replace('auto_model.','')]
        new_state_dict[n] = para + ratio * chat_vector['chat_vector'][n.replace('auto_model.','')]
        # if 'embed_tokens' in n or 'word_embeddings' in n:
        #     if not skip_embed:
        #         assert p.data.shape == chat_vector['chat_vector'][
        #             n].shape, "embeds_token shape mismatch. Use --skip_embed to skip embedding layers."
        #         p.data += ratio * chat_vector['chat_vector'][n]
        #     elif special_tokens_map:
        #         for k, v in special_tokens_map.items():
        #             p.data[k] += ratio * \
        #                 chat_vector['chat_embed'][v]
        # elif 'lm_head' in n:
        #     if not skip_embed:
        #         p.data += ratio * chat_vector['chat_vector'][n]
        #     elif special_tokens_map:
        #         for k, v in special_tokens_map.items():
        #             p.data[k] += ratio * \
        #                 chat_vector['chat_lmhead'][v]
        # else:
            
    base.load_state_dict(new_state_dict, strict=False)
    return base
i = 0
results_dic = {}
for r in [1]:
    for g in [1]:

# Modelli: 
# Funzionano:
# distilbert/distilbert-base-uncased
# nlpie/bio-distilbert-uncased
# sentence-transformers/msmarco-distilbert-base-dot-prod-v3





        extract(base_model_path = 'google-t5/t5-base', chat_model_path='razent/SciFive-base-Pubmed_PMC', output_path = './', l1=r)
        base_model = sentence_models.Transformer('castorini/monot5-base-msmarco')
# base_model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
        base_model = add_chat_vector(base_model, './', g)

        pooling_model = sentence_models.Pooling(768, 'mean')
        s_model = SentenceTransformer(modules=[base_model, pooling_model])

        model = models.SentenceBERT('castorini/monot5-base-msmarco') #sentence-transformers/msmarco-distilbert-base-dot-prod-v3
        model.doc_model = s_model
        model.q_model = s_model

#### Load the SBERT model and retrieve using cosine-similarity
        model = DRES(model, batch_size=16)
        retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
        results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] ,  

        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,3,10])     

        results_dic[i] = {'ratio':r,'gamma':g,'ndcg10':ndcg['NDCG@10']}
        i+=1



model = models.SentenceBERT('razent/SciFive-base-Pubmed_PMC')
        
model = DRES(model, batch_size=16)
retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] ,  
    
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,3,10])     

results_dic[i] = {'ratio':-2,'gamma':-2,'ndcg10':ndcg['NDCG@10']}

model = models.SentenceBERT('castorini/monot5-base-msmarco')
        
model = DRES(model, batch_size=16)
retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] ,  
    
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,3,10])     

results_dic[i+2] = {'ratio':-1,'gamma':-1,'ndcg10':ndcg['NDCG@10']}


print(results_dic)

