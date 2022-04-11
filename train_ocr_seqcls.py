import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup

from utils.focalloss import FocalLoss
from utils.ocr_dataset import make_ocr_seqcls_loader, prep_input

from models.ocr_seqcls import OCR_seqcls_model

def train(args):
    # Constant setup
    NAME = args.NAME
    BATCH_SIZE = args.BATCH_SIZE
    BATCH_SIZE_DEV = args.BATCH_SIZE_DEV
    LR = args.LR
    N_EPOCH = args.N_EPOCH
    GAMMA = args.GAMMA
    ALPHA = args.ALPHA
    SCHEDULER = args.scheduler
    print(f'{NAME} batch_size={BATCH_SIZE}, Adam_lr={LR}, FocalAlpha={ALPHA}, GAMMA={GAMMA}, scheduler={SCHEDULER}\n')

    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    train_loader, tokenizer = make_ocr_seqcls_loader('VisWiz_VQA', 'train', BATCH_SIZE)
    dev_loader,_ = make_ocr_seqcls_loader('VisWiz_VQA', 'val', BATCH_SIZE)

    # Setup Tensorboard
    writer = SummaryWriter(log_dir='./result/runs' ,comment=f'{NAME} batch_size={BATCH_SIZE}, Adam_lr={LR}, FocalAlpha={ALPHA}, GAMMA={GAMMA}')

    # Eval for F1
    def eval(model):
        model.eval()
        with torch.no_grad():
            total_hit, total_pred_positive, total_truth_positive, total_loss, total_pred = 0, 0, 0, [], 0
            for idx, batch in enumerate(dev_loader):
                tokenized, coords_in, cls_mask, outs, img_names = prep_input(batch, tokenizer)

                tokenized = tokenized.to(device)
                coords_in = coords_in.to(device)
                cls_mask = cls_mask.to(device)
                truth = outs.to(device).float().reshape(-1,1)

                pred = model(tokenized, coords_in, cls_mask)

                loss = criterion(pred, truth).detach()

                pred_bin = pred > 0
                truth_bin = truth > 0.5
                
                hit = torch.sum(pred_bin*truth_bin == 1).detach()
                pred_positive = torch.sum(pred > 0).detach()
                truth_positive = torch.sum(truth > 0.5).detach()

                total_loss.append(float(loss))
                total_hit += int(hit)
                total_pred_positive += int(pred_positive)
                total_truth_positive += int(truth_positive)
                total_pred += int(pred.shape[0])
            print('#pred positives',total_pred_positive)
            print('#groundtruth positives',total_truth_positive)
            print('#total pred', total_pred)
            print('#hit', total_hit)
            total_loss = sum(total_loss)/len(total_loss)
            if (total_pred_positive == 0):
                total_pred_positive = 1e10
            prec = total_hit / total_pred_positive
            recall = total_hit / total_truth_positive
            try:
                f1 = 2/(1/prec + 1/recall)
            except:
                f1 = 0
            print('f1', f1)
        return total_loss, prec, recall, f1

    # Training setup
    model = OCR_seqcls_model().to(device)

    criterion = FocalLoss(gamma=GAMMA, alpha=ALPHA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if SCHEDULER == 'linear':
        print('linear scheduler')
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=len(train_loader)*N_EPOCH)

    # Train
    n_iter = 0
    n_prev_iter = 0
    running_loss = 0
    best_f1 = 0
    for epoch in range(N_EPOCH):
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            tokenized, coords_in, cls_mask, outs, img_names = prep_input(batch, tokenizer)

            tokenized = tokenized.to(device)
            coords_in = coords_in.to(device)
            cls_mask = cls_mask.to(device)
            truth = outs.to(device).float().reshape(-1,1)

            pred = model(tokenized, coords_in, cls_mask)

            loss = criterion(pred, truth)

            loss.backward()
            optimizer.step()

            if SCHEDULER:
                scheduler.step()

            n_iter += 1
            writer.add_scalar('Loss/train_batch', loss, n_iter)
            running_loss += loss.detach()

            if batch_idx % 250 == 0:
                print(epoch, batch_idx)
                print(running_loss/(n_iter-n_prev_iter))
                loss, prec, recall, f1 = eval(model)
                writer.add_scalar('Loss/train_avg', running_loss/(n_iter-n_prev_iter), n_iter)
                n_prev_iter = n_iter
                running_loss = 0
                writer.add_scalar('Loss/dev', loss, n_iter)
                writer.add_scalar('Precision/dev', prec, n_iter)
                writer.add_scalar('Recall/dev', recall, n_iter)
                writer.add_scalar('F1/dev', f1, n_iter)

                try:
                    os.makedirs(f'./result/checkpoint/{NAME}')
                except:
                    pass

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save({
                        'epoch': epoch,
                        'step': n_iter,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'dev_loss': loss,
                        }, f'./result/checkpoint/{NAME}/batchsize{BATCH_SIZE}_lr{LR}_FocalALPHA{ALPHA}_GAMMA{GAMMA}_{epoch}_{batch_idx}_{loss}_{f1}.bin')
            break
        break
    print('DONE !!!')
    print(f'Best F1: {best_f1}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--NAME', default='unnamed')
    parser.add_argument('--BATCH_SIZE', default=16, type=int)
    parser.add_argument('--BATCH_SIZE_DEV', default=8, type=int)
    parser.add_argument('--LR', default=5e-6, type=float)
    parser.add_argument('--N_EPOCH', default=30, type=int)
    parser.add_argument('--GAMMA', default=2, type=int)
    parser.add_argument('--ALPHA', default=5, type=int)
    parser.add_argument('--scheduler', default=False)

    args = parser.parse_args()

    train(args)