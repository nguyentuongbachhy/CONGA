@echo off
echo Running CMS with Graph initialization...
python train_sasrec_with_graph.py --dataset ml-1m --train_dir ml-1m_cms --batch_size 64 --lr 0.001 --maxlen 500 --hidden_units 50 --num_blocks 2 --num_epochs 400 --num_heads 1 --dropout_rate 0.2 --num_negatives 9 --device cuda --norm_first --num_workers 2 --inference_only false --state_dict_path None --use_nested_learning true --cms_fast_weight 0.5 --cms_medium_weight 0.3 --cms_slow_weight 0.2

echo.
echo Running CMS baseline (random initialization)...
python main.py --dataset ml-1m --train_dir ml-1m_cms_baseline --batch_size 64 --lr 0.001 --maxlen 500 --hidden_units 50 --num_blocks 2 --num_epochs 400 --num_heads 1 --dropout_rate 0.2 --num_negatives 9 --device cuda --norm_first --num_workers 2 --inference_only false --state_dict_path None --use_nested_learning true --cms_fast_weight 0.5 --cms_medium_weight 0.3 --cms_slow_weight 0.2

echo.
echo Both experiments completed!
pause
