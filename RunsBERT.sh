# RunsBERT

#python3 Sentence_bert.py --dataset 'scifact' --device 'cuda:1' --mod 'cos' --model_name 'MiniLM_roberta' --model_base_path 'sentence-transformers/msmarco-roberta-base-v3' --model_vector_plus_path 'allenai/biomed_roberta_base' --model_vector_minus_path 'FacebookAI/roberta-base'   

#python3 Sentence_bert.py --dataset 'nfcorpus' --device 'cuda:1' --mod 'cos' --model_name 'MiniLM_roberta' --model_base_path 'sentence-transformers/msmarco-roberta-base-v3' --model_vector_plus_path 'allenai/biomed_roberta_base' --model_vector_minus_path 'FacebookAI/roberta-base'   

#####################à Ablation
# python3 run_model_sum_weights.py --dataset 'trec-covid' --ablation 'remove' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

# python3 run_model_sum_weights.py --dataset 'trec-covid' --ablation 'add' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

# python3 run_model_sum_weights.py --dataset 'nfcorpus' --ablation 'remove' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

# python3 run_model_sum_weights.py --dataset 'nfcorpus' --ablation 'add' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

# python3 run_model_sum_weights.py --dataset 'scifact' --ablation 'remove' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

# python3 run_model_sum_weights.py --dataset 'scifact' --ablation 'add' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

# python3 run_model_sum_weights.py --dataset 'scidocs' --ablation 'remove' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

#python3 run_model_sum_weights.py --dataset 'scidocs' --ablation 'add' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

python3 run_model_sum_weights.py --dataset 'trec-covid' --ablation 'add' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'trec-covid' --ablation 'remove' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'scifact' --ablation 'add' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'scifact' --ablation 'remove' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'nfcorpus' --ablation 'add' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'nfcorpus' --ablation 'remove' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'scidocs' --ablation 'add' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'scidocs' --ablation 'remove' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   




#########################
# python3 run_model_sum_weights.py --dataset 'scifact' --alfa 0.3 --device 'cuda:1' --mod 'cos' --model_name 'MiniLM_roberta' --model_base_path 'sentence-transformers/msmarco-roberta-base-v3' --model_vector_plus_path 'allenai/biomed_roberta_base' --model_vector_minus_path 'FacebookAI/roberta-base'   

# python3 run_model_sum_weights.py --dataset 'nfcorpus' --alfa 0.3 --device 'cuda:1' --mod 'cos' --model_name 'MiniLM_roberta' --model_base_path 'sentence-transformers/msmarco-roberta-base-v3' --model_vector_plus_path 'allenai/biomed_roberta_base' --model_vector_minus_path 'FacebookAI/roberta-base'   


# python3 run_model_sum_weights.py --dataset 'scifact' --alfa 0.4 --device 'cuda:1' --mod 'cos' --model_name 'MiniLM_roberta' --model_base_path 'sentence-transformers/msmarco-roberta-base-v3' --model_vector_plus_path 'allenai/biomed_roberta_base' --model_vector_minus_path 'FacebookAI/roberta-base'   

# python3 run_model_sum_weights.py --dataset 'nfcorpus' --alfa 0.4 --device 'cuda:1' --mod 'cos' --model_name 'MiniLM_roberta' --model_base_path 'sentence-transformers/msmarco-roberta-base-v3' --model_vector_plus_path 'allenai/biomed_roberta_base' --model_vector_minus_path 'FacebookAI/roberta-base'   

# python3 run_model_sum_weights.py --dataset 'scifact' --alfa 0.4 --device 'cuda:1' --mod 'cos' --model_name 'MiniLM_roberta' --model_base_path 'sentence-transformers/msmarco-roberta-base-v3' --model_vector_plus_path 'allenai/biomed_roberta_base' --model_vector_minus_path 'FacebookAI/roberta-base'   

# python3 run_model_sum_weights.py --dataset 'nfcorpus' --alfa 0.4 --device 'cuda:1' --mod 'cos' --model_name 'MiniLM_roberta' --model_base_path 'sentence-transformers/msmarco-roberta-base-v3' --model_vector_plus_path 'allenai/biomed_roberta_base' --model_vector_minus_path 'FacebookAI/roberta-base'   




# BASE Scientific

#python3 run_model_sum_weights.py --alfa 0.2 --dataset 'dbpedia-entity' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'Suchinthana/T5-Base-Wikigen' --model_vector_minus_path 'google-t5/t5-base'

#python3 run_model_sum_weights.py --alfa 0.4 --dataset 'dbpedia-entity' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'Suchinthana/T5-Base-Wikigen' --model_vector_minus_path 'google-t5/t5-base'

#python3 run_model_sum_weights.py --alfa 0.1 --dataset 'fiqa' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'xuan8888888/t5-base-financial-title-generation' --model_vector_minus_path 'google-t5/t5-base'

# python3 run_model_sum_weights.py --alfa 0.7 --dataset 'trec-covid' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'

# python3 run_model_sum_weights.py --alfa 0.7 --dataset 'scidocs' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'

# # BASE multilingual

# python3 run_model_sum_weights.py --alfa 0.7 --dataset 'germanquad' --model_name 't5_base' --model_base_path 'unicamp-dl/mt5-base-en-msmarco' --model_vector_plus_path 'airKlizz/mt5-base-wikinewssum-german' --model_vector_minus_path 'google/mt5-base'

# python3 run_model_sum_weights.py --alfa 0.7 --dataset 'germanquad' --model_name 't5_base' --model_base_path 'unicamp-dl/mt5-base-en-msmarco' --model_vector_plus_path 'airKlizz/mt5-base-wikinewssum-german' --model_vector_minus_path 'airKlizz/mt5-base-wikinewssum-english'

# MINILM Scientific

# runaksh/financial_summary_T5_base

# python3 run_model_sum_weights.py --alfa 0.1 --device 'cuda:1' --mod 'cos' --dataset 'nfcorpus' --model_name 'MiniLM' --model_base_path 'sentence-transformers/all-MiniLM-L6-v2' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'nreimers/MiniLM-L6-H384-uncased'
# # python3 run_model_sum_weights.py --alfa 0.5 --device 'cuda:1' --mod 'cos' --dataset 'nfcorpus' --model_name 'MiniLM' --model_base_path 'sentence-transformers/msmarco-MiniLM-L6-cos-v5' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

# python3 run_model_sum_weights.py --alfa 0.1 --device 'cuda:1' --mod 'cos' --dataset 'scifact' --model_name 'MiniLM' --model_base_path 'sentence-transformers/all-MiniLM-L6-v2' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'nreimers/MiniLM-L6-H384-uncased'
# # python3 run_model_sum_weights.py --alfa 0.5 --device 'cuda:1' --mod 'cos' --dataset 'scifact' --model_name 'MiniLM' --model_base_path 'sentence-transformers/msmarco-MiniLM-L6-cos-v5' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

# python3 run_model_sum_weights.py --alfa 0.1 --device 'cuda:1' --mod 'cos' --dataset 'scidocs' --model_name 'MiniLM' --model_base_path 'sentence-transformers/all-MiniLM-L6-v2' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'nreimers/MiniLM-L6-H384-uncased'
# # python3 run_model_sum_weights.py --alfa 0.5 --device 'cuda:1' --mod 'cos' --dataset 'scidocs' --model_name 'MiniLM' --model_base_path 'sentence-transformers/msmarco-MiniLM-L6-cos-v5' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

# python3 run_model_sum_weights.py --alfa 0.1 --device 'cuda:1' --mod 'cos' --dataset 'trec-covid' --model_name 'MiniLM' --model_base_path 'sentence-transformers/all-MiniLM-L6-v2' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'nreimers/MiniLM-L6-H384-uncased'
# # python3 run_model_sum_weights.py --alfa 0.5 --device 'cuda:1' --mod 'cos' --dataset 'trec-covid' --model_name 'MiniLM' --model_base_path 'sentence-transformers/msmarco-MiniLM-L6-cos-v5' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

# # python3 run_model_sum_weights.py --alfa 0.1 --dataset 'trec-covid' --model_name 'MiniLM' --model_base_path 'cross-encoder/ms-marco-MiniLM-L-6-v2' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

# # python3 run_model_sum_weights.py --alfa 0.1 --dataset 'scidocs' --model_name 'MiniLM' --model_base_path 'cross-encoder/ms-marco-MiniLM-L-6-v2' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

# python3 Value_lambda.py --dataset 'nfcorpus' --device 'cuda:1' --model_name 'MiniLM' --model_base_path 'cross-encoder/ms-marco-MiniLM-L-6-v2' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

# python3 Value_lambda.py --dataset 'scifact' --device 'cuda:1' --model_name 'MiniLM' --model_base_path 'cross-encoder/ms-marco-MiniLM-L-6-v2' --model_vector_plus_path 'menadsa/BioS-MiniLM' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

# python3 Value_lambda.py --dataset 'nfcorpus' --device 'cuda:1' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'

# python3 Value_lambda.py --dataset 'scifact' --device 'cuda:1' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'  #kamalkraj/bioelectra-base-discriminator-pubmed


#python3 Value_lambda.py --device 'cuda:0' --dataset 'nfcorpus' --model_name 'MiniLM_electra' --model_base_path 'cross-encoder/ms-marco-electra-base' --model_vector_plus_path 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc' --model_vector_minus_path 'google/electra-base-discriminator'

#python3 Value_lambda.py --device 'cuda:0' --dataset 'scifact' --model_name 'MiniLM_electra' --model_base_path 'cross-encoder/ms-marco-electra-base' --model_vector_plus_path 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc' --model_vector_minus_path 'google/electra-base-discriminator'

# WikinewsSum/t5-base-multi-en-wiki-news
# legacy107/ms-marco-MiniLM-L-6-v2-wikipedia-augmented-search-farmed

#python3 run_model_sum_weights.py --device 'cuda:1' --alfa 0.7  --dataset 'nfcorpus' --model_name 'MiniLM_electra_pmc' --model_base_path 'cross-encoder/ms-marco-electra-base' --model_vector_plus_path 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc' --model_vector_minus_path 'google/electra-base-discriminator'

#python3 run_model_sum_weights.py --device 'cuda:1' --alfa 0.7  --dataset 'scifact' --model_name 'MiniLM_electra_pmc' --model_base_path 'cross-encoder/ms-marco-electra-base' --model_vector_plus_path 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc' --model_vector_minus_path 'google/electra-base-discriminator'

#python3 run_model_sum_weights.py --device 'cuda:1' --alfa 0.7  --dataset 'trec-covid' --model_name 'MiniLM_electra_pmc'  --model_base_path 'cross-encoder/ms-marco-electra-base' --model_vector_plus_path 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc' --model_vector_minus_path 'google/electra-base-discriminator'

#python3 run_model_sum_weights.py --device 'cuda:1' --alfa 0.7 --dataset 'scidocs' --model_name 'MiniLM_electra_pmc'  --model_base_path 'cross-encoder/ms-marco-electra-base' --model_vector_plus_path 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc' --model_vector_minus_path 'google/electra-base-discriminator'

# p10 -> electra wiki, RunsBERT

# runt5 -> T5 base wikipedia, RunsT5

# python3 run_model_sum_weights.py --dataset 'dbpedia-entity' --alfa 0.1 --device 'cuda:1' --model_name 'MiniLM' --model_base_path 'cross-encoder/ms-marco-MiniLM-L-6-v2' --model_vector_plus_path 'legacy107/ms-marco-MiniLM-L-6-v2-wikipedia-augmented-search-farmed' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

# python3 run_model_sum_weights.py --dataset 'dbpedia-entity' --alfa 0.2 --device 'cuda:1' --model_name 'MiniLM' --model_base_path 'cross-encoder/ms-marco-MiniLM-L-6-v2' --model_vector_plus_path 'legacy107/ms-marco-MiniLM-L-6-v2-wikipedia-augmented-search-farmed' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'


# python3 run_model_sum_weights.py --dataset 'climate-fever' --alfa 0.1 --device 'cuda:1' --model_name 'MiniLM' --model_base_path 'cross-encoder/ms-marco-MiniLM-L-6-v2' --model_vector_plus_path 'legacy107/ms-marco-MiniLM-L-6-v2-wikipedia-augmented-search-farmed' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

# python3 run_model_sum_weights.py --dataset 'climate-fever' --alfa 0.2 --device 'cuda:1' --model_name 'MiniLM' --model_base_path 'cross-encoder/ms-marco-MiniLM-L-6-v2' --model_vector_plus_path 'legacy107/ms-marco-MiniLM-L-6-v2-wikipedia-augmented-search-farmed' --model_vector_minus_path 'sentence-transformers/all-MiniLM-L6-v2'

#python3 run_model_sum_weights.py --dataset 'trec-covid' --alfa 0.7 --device 'cuda:1' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'

#python3 run_model_sum_weights.py --dataset 'trec-covid' --alfa 0.9 --device 'cuda:0' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'

#python3 run_model_sum_weights.py --dataset 'scidocs' --alfa 0.7 --device 'cuda:1' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'

#python3 run_model_sum_weights.py --dataset 'scifact' --alfa 0.7 --device 'cuda:1' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'
