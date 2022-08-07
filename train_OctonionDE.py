import config
from models import *
import json
import os 

'''best setting 1 for WN18RR -> output1_WN18RR.txt
con = config.Config()
con.set_in_path("./benchmarks/WN18RR/")
con.set_work_threads(8)
con.set_train_times(20000)
con.set_nbatches(10)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(100) 
con.set_lmbda(0.1)
con.set_lmbda_two(0.01)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1000)
con.set_valid_steps(1000)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 2 for WN18RR -> output2_WN18RR.txt
con = config.Config()
con.set_in_path("./benchmarks/WN18RR/")
con.set_work_threads(8)
con.set_train_times(10000)
con.set_nbatches(100)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(100)
con.set_lmbda(0.1)
con.set_lmbda_two(0.01)
con.set_margin(1.0)
con.set_ent_neg_rate(10)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1000)
con.set_valid_steps(1000)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint_WN18RR")
con.set_result_dir("./result_WN18RR")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 3 for WN18RR -> output3_WN18RR.txt
con = config.Config()
con.set_in_path("./benchmarks/WN18RR/")
con.set_work_threads(8)
con.set_train_times(10000)
con.set_nbatches(10)
con.set_alpha(0.022)
con.set_bern(1)
con.set_dimension(150) 
con.set_lmbda(0.25)
con.set_lmbda_two(0.25)
con.set_margin(1.0)
con.set_ent_neg_rate(2)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1000)
con.set_valid_steps(1000)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 1 for FB15k237 -> output1_FB15k237.txt
con = config.Config()
con.set_in_path("./benchmarks/FB15K237/")
con.set_work_threads(8)
con.set_train_times(5000)
con.set_nbatches(10)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(100)
con.set_lmbda(0.3)
con.set_lmbda_two(0.3)
con.set_margin(1.0)
con.set_ent_neg_rate(10)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1000)
con.set_valid_steps(1000)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 2 for FB15k237 -> output2_FB15k237.txt
con = config.Config()
con.set_in_path("./benchmarks/FB15K237/")
con.set_work_threads(8)
con.set_train_times(9000) 
con.set_nbatches(10)
con.set_alpha(0.02)
con.set_bern(1)
con.set_dimension(100)
con.set_lmbda(0.1)
con.set_lmbda_two(0.1)
con.set_margin(1.0)
con.set_ent_neg_rate(10)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(3000) 
con.set_valid_steps(3000) 
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 3 for FB15k237 -> output3_FB15k237.txt
con = config.Config()
con.set_in_path("./benchmarks/FB15K237/")
con.set_work_threads(8)
con.set_train_times(4000) 
con.set_nbatches(100)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(300)
con.set_lmbda(0.3)
con.set_lmbda_two(0.3)
con.set_margin(1.0)
con.set_ent_neg_rate(10)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1000) 
con.set_valid_steps(1000)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 1 for WN18 -> output1_WN18.txt
con = config.Config()
con.set_in_path("./benchmarks/WN18/")
con.set_work_threads(8)
con.set_train_times(6000) 
con.set_nbatches(10)  
con.set_alpha(0.035)
con.set_bern(1)
con.set_dimension(200)
con.set_lmbda(0.03) 
con.set_margin(1.0)
con.set_ent_neg_rate(10)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1000) 
con.set_valid_steps(1000) 
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 2 for WN18 -> output2_WN18.txt
con = config.Config()
con.set_in_path("./benchmarks/WN18/")
con.set_work_threads(8)
con.set_train_times(3000) 
con.set_nbatches(100)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(250) 
con.set_lmbda(0.05)
con.set_lmbda_two(0.05)
con.set_margin(1.0)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(300)
con.set_valid_steps(300)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 3 for WN18 -> output3_WN18.txt
con = config.Config()
con.set_in_path("./benchmarks/WN18/")
con.set_work_threads(8)
con.set_train_times(5000) 
con.set_nbatches(10)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(200) 
con.set_lmbda(0.05)
con.set_lmbda_two(0.05)
con.set_margin(1.0)
con.set_ent_neg_rate(10)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1000) 
con.set_valid_steps(1000) 
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 1 for FB15k -> output1_FB15k.txt
con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_work_threads(8)
con.set_train_times(20000) 
con.set_nbatches(10)
con.set_alpha(0.02)
con.set_bern(1)
con.set_dimension(50) #best 50, 100 not enough memory cuda
con.set_lmbda(0.03)
con.set_lmbda_two(0)
con.set_margin(1.0)
con.set_ent_neg_rate(10)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1000) 
con.set_valid_steps(1000) 
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 2 for FB15k -> output2_FB15k.txt but need change dim >= 300 
con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_work_threads(8)
con.set_train_times(10000) 
con.set_nbatches(100)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(250) 
con.set_lmbda(0.05)
con.set_lmbda_two(0.05)
con.set_margin(1.0)
con.set_ent_neg_rate(5)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(1000) 
con.set_valid_steps(1000) 
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''

'''best setting 3 for FB15k -> output3_FB15k.txt but need change dim >= 150 
con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_work_threads(8)
con.set_train_times(20000) 
con.set_nbatches(10)
con.set_alpha(0.1)
con.set_bern(1)
con.set_dimension(200) #best 100
con.set_lmbda(0.05)
con.set_lmbda_two(0.05)
con.set_margin(1.0)
con.set_ent_neg_rate(7)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(4000) 
con.set_valid_steps(4000) 
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(OctonionDE)
con.train()
'''