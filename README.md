# VOMP
Video regression Of Multi-People

test - lihaoyuan

# citing 
ROMP : https://github.com/Arthur151/ROMP

# should konw
1. at transformer.py line:98 -> attn_mask should be send to cuda if device=cuda

# modify
1. change the input or output of transformer.py
2. then change the input of vompv1 which uses tansformer to handle feature 
3. at last change the input methodes of base model in base.py -> line:68
4. check whether you shoule change the parameter of transformer.gourp in congfig.py

test