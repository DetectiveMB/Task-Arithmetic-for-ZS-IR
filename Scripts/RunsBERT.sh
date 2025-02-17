# RunsBERT


##################### Ablation
python3 run_model_sum_weights.py --dataset 'trec-covid' --ablation 'remove' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

python3 run_model_sum_weights.py --dataset 'trec-covid' --ablation 'add' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

python3 run_model_sum_weights.py --dataset 'nfcorpus' --ablation 'remove' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

python3 run_model_sum_weights.py --dataset 'nfcorpus' --ablation 'add' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

python3 run_model_sum_weights.py --dataset 'scifact' --ablation 'remove' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

python3 run_model_sum_weights.py --dataset 'scifact' --ablation 'add' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

python3 run_model_sum_weights.py --dataset 'scidocs' --ablation 'remove' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

python3 run_model_sum_weights.py --dataset 'scidocs' --ablation 'add' --alfa 0.7 --device 'cuda:1' --mod 'cos' --model_name 't5_base' --model_base_path 'castorini/monot5-base-msmarco' --model_vector_plus_path 'razent/SciFive-base-Pubmed_PMC' --model_vector_minus_path 'google-t5/t5-base'   

python3 run_model_sum_weights.py --dataset 'trec-covid' --ablation 'add' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'trec-covid' --ablation 'remove' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'scifact' --ablation 'add' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'scifact' --ablation 'remove' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'nfcorpus' --ablation 'add' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'nfcorpus' --ablation 'remove' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'scidocs' --ablation 'add' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   

python3 run_model_sum_weights.py --dataset 'scidocs' --ablation 'remove' --alfa 0.8 --device 'cuda:1' --mod 'cos' --model_name 'Llama-2-7b' --model_base_path 'zyznull/RankingGPT-llama2-7b' --model_vector_plus_path 'nlpie/Llama2-MedTuned-7b' --model_vector_minus_path 'meta-llama/Llama-2-7b-hf'   




