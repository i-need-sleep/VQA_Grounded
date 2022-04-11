from winreg import REG_OPTION_BACKUP_RESTORE
import torch
from transformers import RobertaModel

class OCR_seqcls_model(torch.nn.Module):
    def __init__(self):
        super(OCR_seqcls_model, self).__init__()

        self.roberta = RobertaModel.from_pretrained('./cache/roberta_large')
        self.lin_coord = torch.nn.Linear(8, 1024)
        self.clsHead = torch.nn.Linear(1024, 1)

    def forward(self, tokenized, coords_in, cls_mask):
        txt_embs = self.roberta.embeddings(tokenized)
        coord_embs = self.lin_coord(coords_in)
        out = self.roberta.encoder(txt_embs + coord_embs).last_hidden_state
        out = self.clsHead(out)
        out = out[cls_mask].reshape(-1,1)
        return out