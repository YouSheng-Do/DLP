# best one
# python Trainer.py --DR ../LAB4_Dataset/LAB4_Dataset/ --save_root ../lab4_checkpoint --optim Adam --per_save 1 --store_visualization --fast_train --kl_anneal_ratio 0.9 --tfr_d_step 0.1 --kl_anneal_type Cyclical
# python Trainer.py --DR ../LAB4_Dataset/LAB4_Dataset/ --save_root ../lab4_checkpoint --optim Adam --per_save 1 --store_visualization --fast_train --kl_anneal_ratio 0.5 --tfr_d_step 0.1 --kl_anneal_type Monotonic --kl_anneal_cycle 1
# python Trainer.py --DR ../LAB4_Dataset/LAB4_Dataset/ --save_root ../lab4_checkpoint --optim Adam --per_save 1 --store_visualization --fast_train --tfr_d_step 0.1 --kl_anneal_type None 
python Trainer.py --DR ../LAB4_Dataset/LAB4_Dataset/ --save_root ../lab4_checkpoint --optim Adam --per_save 1 --store_visualization --tfr_d_step 0.1 --kl_anneal_type None --ckpt_path ../lab4_checkpoint/epoch=69.ckpt --num_epoch 10