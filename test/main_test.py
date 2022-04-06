import sys 
sys.path.append("..")
import unittest
import os
import os.path as osp
from dl.main import main
from dl import args


class MyTestCase(unittest.TestCase):
    def test_main(self):
        main_script_dir = "../dl/"
        os.chdir(osp.dirname(main_script_dir))
        args.max_epochs = 1
        #main(args)

    def test_eval(self):
        main_script_dir = "../dl/"
        os.chdir(osp.dirname(main_script_dir))
        args.model = "pdglstm"    # test model
        args.num_gpus = 1           
        args.model_num = 30       # test model number
        args.mode = "eval"        
        args.graph_info = True
        args.demo = True     # True if use provided trained models
        #args.permutation_test = True    # True if generate permutation importance records
        #args.mis_aly = True     # True if generate correctly and incorrectly node files
        if args.graph_info:
            args.batch_size = 32
        else:
            args.batch_size = 32000
        main(args)


if __name__ == '__main__':
    unittest.main()
