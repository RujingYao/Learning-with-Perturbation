# Learning with Perturbation: Taxonomy and Algorithms
This is the code for Learning with Perturbation: Taxonomy and Algorithms.

## Environment 

    pip install -r requirement.txt
    
### ***Image classification***  


+ 2.1 Parameters  

    | Parameters         | Type               | Optional                | 
    | ---- | ----------: | ------------------: | --------------------: | 
    |  epoch             |       int   |  defult:150              |        
    |  batch_size        |       int   |  defult:128              |           
    |  lr                |     float   |  defult:1e-2           |      
    |  soft              |       bool  |  True False   defult:True      |   
    |  dataset           |       str   |  cifar10, cifar100        |       
    |  noise             |      bool   |  True False             |    
    |  ntype             |       str   |  random pair  effective when noise=True |      
    |  nrate             |       str   |  0.1 0.2 0.3  effective when noise=True       |      


+ 2.2 Training file introduction 

       image_compensation/train_ori.py                # the script of base method
       image_compensation/train_logit_l1_comp.py      # logit-l1 compensation
       image_compensation/train_label_mix_comp.py     # Mixed-label compensation
