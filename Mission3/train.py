"""Portable reference for the reported focused-positive-weight BCE experiments.
Fresh training requires an independently verified provided-data list and local public pretrained weights.
This module does not launch training during inference or notebook record review.
"""
import argparse,csv,json,math,os,random,time
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from model import LABELS,POSITIVE_WEIGHTS,MAX_LENGTH,THRESHOLD,text_only,probabilities

class Rows(Dataset):
    def __init__(self,texts,y,tok):self.texts=texts;self.y=y;self.tok=tok
    def __len__(self):return len(self.texts)
    def __getitem__(self,i):return dict(self.tok(self.texts[i],truncation=True,max_length=MAX_LENGTH),labels=self.y[i].astype('float32').tolist())

def read_data(path):
    texts,labels=[],[];path=Path(path)
    if path.is_file() and path.suffix=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                texts.append(str(row.get('text') or ''));labels.append([int(float(row[l])) for l in LABELS])
    else:
        for file in sorted(path.rglob('*.json')):
            raw=json.loads(file.read_text(encoding='utf-8-sig'));texts.append(text_only(raw))
            targets=raw.get('symptom')
            if targets is None:targets=raw.get('metadata',{}).get('symptom',[])
            if isinstance(targets,str):targets=[targets]
            labels.append([int(l in (targets or [])) for l in LABELS])
    if not texts:raise ValueError('No data')
    # No sample is removed based on symptoms. Only labels outside LABELS are ignored.
    return texts,np.asarray(labels,dtype=np.float32)

@torch.no_grad()
def evaluate(model,dl,device):
    model.eval();ps=[];ys=[]
    for batch in dl:
        ys.append(batch.pop('labels').numpy());ps.append(probabilities(model,batch,device).numpy())
    p=np.concatenate(ps);y=np.concatenate(ys)
    return float(f1_score(y,p>=THRESHOLD,average='macro',zero_division=0)),p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train_data',type=Path,required=True,help='Verified supplied Seoul training JSON folder or CSV')
    ap.add_argument('--validation_data',type=Path,required=True,help='Verified supplied Validation JSON folder or CSV; evaluation only')
    ap.add_argument('--pretrained_dir',type=Path,required=True,help='Fresh public klue/roberta-base stored locally')
    ap.add_argument('--output_dir',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True,help='Set explicitly; source notebook code/output disagree on seed')
    args=ap.parse_args()
    if args.train_data.resolve()==args.validation_data.resolve():raise ValueError('Training and Validation must be separate')
    texts,y=read_data(args.train_data);vt,vy=read_data(args.validation_data)
    if (len(texts),len(vt))!=(29200,3640):raise ValueError('Reported experiment expects29200/3640 supplied files; do not filter labels to force these counts')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(6);device='cuda' if torch.cuda.is_available() else 'cpu'
    tok=AutoTokenizer.from_pretrained(args.pretrained_dir,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(args.pretrained_dir,num_labels=9,problem_type='multi_label_classification',local_files_only=True).to(device)
    collate=DataCollatorWithPadding(tok,pad_to_multiple_of=8)
    td=DataLoader(Rows(texts,y,tok),batch_size=8,shuffle=True,collate_fn=collate)
    vd=DataLoader(Rows(vt,vy,tok),batch_size=16,shuffle=False,collate_fn=collate)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    steps=math.ceil(len(td)/2);schedule=get_linear_schedule_with_warmup(opt,int(steps*8*.1),steps*8)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POSITIVE_WEIGHTS,device=device))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    history=[];best=-1.;bad=0
    for epoch in range(1,9):
        model.train();opt.zero_grad();losses=0.;started=time.perf_counter()
        for i,batch in enumerate(td):
            target=batch.pop('labels').to(device)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                logits=model(**{k:v.to(device) for k,v in batch.items()}).logits
                raw_loss=loss_fn(logits,target);loss=raw_loss/2
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();losses+=raw_loss.item()
            if (i+1)%2==0 or i+1==len(td):
                scaler.step(opt);scaler.update();schedule.step();opt.zero_grad()
        score,p=evaluate(model,vd,device)
        row={'epoch':epoch,'train_loss':losses/len(td),'macro_f1':score,'seconds':time.perf_counter()-started,'threshold':THRESHOLD,'seed':args.seed}
        history.append(row);(args.output_dir/'history.json').write_text(json.dumps(history,indent=2));print(row,flush=True)
        if score>best:
            best=score;bad=0;dest=args.output_dir/'model';dest.mkdir(exist_ok=True)
            model.config.save_pretrained(dest);tok.save_pretrained(dest);torch.save(model.state_dict(),dest/'model.pt')
            np.save(args.output_dir/'validation_probabilities.npy',p);(args.output_dir/'best.json').write_text(json.dumps(row,indent=2))
        else:bad+=1
        if bad>=2:break
if __name__=='__main__':main()
