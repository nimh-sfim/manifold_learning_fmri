import numpy as np

def scan_level_split():
    i = 1
    while i<=4:
        if i == 1:
            train_idx  = np.arange(364)
            test_idx   = np.arange(365,729)
        if i == 2:
            train_idx  = np.arange(365,729)
            test_idx   = np.arange(364)
        if i == 3:
            train_idx = np.concatenate([np.arange(45),np.arange(91,91+45),np.arange(182,182+45),np.arange(273,273+45),np.arange(364,364+45),np.arange(455,455+45),np.arange(546,546+45),np.arange(637,637+45)])
            test_idx  = np.concatenate([np.arange(45,45+46),np.arange(136,136+46),np.arange(227,227+46),np.arange(318,318+46),np.arange(409,409+46),np.arange(500,500+46),np.arange(591,591+46),np.arange(682,682+47)])
        if i == 4:
            train_idx = np.concatenate([np.arange(45,45+46),np.arange(136,136+46),np.arange(227,227+46),np.arange(318,318+46),np.arange(409,409+46),np.arange(500,500+46),np.arange(591,591+46),np.arange(682,682+47)])
            test_idx  = np.concatenate([np.arange(45),np.arange(91,91+45),np.arange(182,182+45),np.arange(273,273+45),np.arange(364,364+45),np.arange(455,455+45),np.arange(546,546+45),np.arange(637,637+45)])
        yield train_idx, test_idx
        i += 1

def group_level_split():
    i = 1
    while i <=2:
        if i == 1:
           aux_train_idx = [np.arange(364)+s*729 for s in range(20)]
           train_idx     = np.array(aux_train_idx).flatten()
           aux_test_idx  = [np.arange(365,729)+s*729 for s in range(20)]
           test_idx      = np.array(aux_test_idx).flatten()
        if i == 2:
           aux_train_idx  = [np.arange(365,729)+s*729 for s in range(20)]
           train_idx      = np.array(aux_train_idx).flatten()
           aux_test_idx   = [np.arange(364)+s*729 for s in range(20)]
           test_idx       = np.array(aux_test_idx).flatten()
        yield train_idx, test_idx
        i += 1
           