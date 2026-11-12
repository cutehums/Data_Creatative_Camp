"""Official single-command offline inference. No labels enter the prediction pipeline."""
import argparse,csv,json,os,platform,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer,DataCollatorWithPadding
from model import LABELS,THRESHOLD,MAX_LENGTH,text_only,load_classifier,probabilities

class TextRows(Dataset):
    def __init__(self,texts,tokenizer):self.texts=texts;self.tokenizer=tokenizer
    def __len__(self):return len(self.texts)
    def __getitem__(self,i):return self.tokenizer(self.texts[i],truncation=True,max_length=MAX_LENGTH,padding=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audio_dir',required=True,help='Official argument; Mission3 does not read audio')
    ap.add_argument('--label_dir',type=Path,required=True)
    ap.add_argument('--ckpt_path',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--batch_size',type=int,default=16)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args();started=time.perf_counter();torch.set_num_threads(6)
    if args.batch_size<1:raise ValueError('batch_size must be positive')
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    model,root=load_classifier(args.ckpt_path,device)
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
    files=sorted(args.label_dir.rglob('*.json'))
    if not files or len({p.name for p in files})!=len(files):raise ValueError('Expected nonempty JSON folder with unique file names')
    texts=[text_only(json.loads(p.read_text(encoding='utf-8-sig'))) for p in files]
    dl=DataLoader(TextRows(texts,tokenizer),batch_size=args.batch_size,shuffle=False,num_workers=0,
                  pin_memory=device=='cuda',collate_fn=DataCollatorWithPadding(tokenizer,padding=True,pad_to_multiple_of=8))
    forward_started=time.perf_counter();ps=[probabilities(model,batch,device) for batch in dl]
    predictions=torch.cat(ps).numpy()>=THRESHOLD;forward_seconds=time.perf_counter()-forward_started
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['label file name','symptom'])
        for path,row in zip(files,predictions.tolist()):
            writer.writerow([path.name,json.dumps([label for label,yes in zip(LABELS,row) if yes],ensure_ascii=False)])
    total=sum(p.numel() for p in model.parameters());elapsed=time.perf_counter()-started
    report={'samples':len(files),'batch_size':args.batch_size,'device':device,
            'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
            'python':platform.python_version(),'torch':torch.__version__,
            'total_parameters':total,'active_parameters':total,'threshold':THRESHOLD,
            'forward_with_tokenization_seconds':forward_seconds,'total_seconds':elapsed,
            'seconds_per_sample':elapsed/len(files),'precision':'CUDA FP16 autocast; CPU float32'}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
