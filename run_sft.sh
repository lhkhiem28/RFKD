python train_sft.py --llm_name 'gemma-2-2b' --run_name "wsft_0.9" --gamma 0.9
python test.py      --llm_name 'gemma-2-2b' --run_name "wsft_0.9" --gamma 0.9 --checkpoint_path output/train/DrugADMET/llm_gemma-2-2b_train_8_wsft_0.9.pth
python train_sft.py --llm_name 'granite-3.3-2b' --run_name "wsft_0.9" --gamma 0.9
python test.py      --llm_name 'granite-3.3-2b' --run_name "wsft_0.9" --gamma 0.9 --checkpoint_path output/train/DrugADMET/llm_granite-3.3-2b_train_8_wsft_0.9.pth
python train_sft.py --llm_name 'gemma-2-2b' --run_name "wsft_0.8" --gamma 0.8
python test.py      --llm_name 'gemma-2-2b' --run_name "wsft_0.8" --gamma 0.8 --checkpoint_path output/train/DrugADMET/llm_gemma-2-2b_train_8_wsft_0.8.pth
python train_sft.py --llm_name 'granite-3.3-2b' --run_name "wsft_0.8" --gamma 0.8
python test.py      --llm_name 'granite-3.3-2b' --run_name "wsft_0.8" --gamma 0.8 --checkpoint_path output/train/DrugADMET/llm_granite-3.3-2b_train_8_wsft_0.8.pth
python train_sft.py --llm_name 'gemma-2-2b' --run_name "wsft_0.7" --gamma 0.7
python test.py      --llm_name 'gemma-2-2b' --run_name "wsft_0.7" --gamma 0.7 --checkpoint_path output/train/DrugADMET/llm_gemma-2-2b_train_8_wsft_0.7.pth
python train_sft.py --llm_name 'granite-3.3-2b' --run_name "wsft_0.7" --gamma 0.7
python test.py      --llm_name 'granite-3.3-2b' --run_name "wsft_0.7" --gamma 0.7 --checkpoint_path output/train/DrugADMET/llm_granite-3.3-2b_train_8_wsft_0.7.pth
python train_sft.py --llm_name 'gemma-2-2b' --run_name "wsft_0.6" --gamma 0.6
python test.py      --llm_name 'gemma-2-2b' --run_name "wsft_0.6" --gamma 0.6 --checkpoint_path output/train/DrugADMET/llm_gemma-2-2b_train_8_wsft_0.6.pth
python train_sft.py --llm_name 'granite-3.3-2b' --run_name "wsft_0.6" --gamma 0.6
python test.py      --llm_name 'granite-3.3-2b' --run_name "wsft_0.6" --gamma 0.6 --checkpoint_path output/train/DrugADMET/llm_granite-3.3-2b_train_8_wsft_0.6.pth
python train_sft.py --llm_name 'gemma-2-2b' --run_name "wsft_0.5" --gamma 0.5
python test.py      --llm_name 'gemma-2-2b' --run_name "wsft_0.5" --gamma 0.5 --checkpoint_path output/train/DrugADMET/llm_gemma-2-2b_train_8_wsft_0.5.pth
python train_sft.py --llm_name 'granite-3.3-2b' --run_name "wsft_0.5" --gamma 0.5
python test.py      --llm_name 'granite-3.3-2b' --run_name "wsft_0.5" --gamma 0.5 --checkpoint_path output/train/DrugADMET/llm_granite-3.3-2b_train_8_wsft_0.5.pth