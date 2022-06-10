class yaml():
    def __init__(self) -> None:
        self.ar = 1
        self.emb_size = 256
        self.trans_heads = 8
        self.forward_dim = 1024
        self.trans_layer = 4 // 2
        self.src_len = 64
        self.tgt_len = 64
        self.dk = 64

def args():
    return yaml()
