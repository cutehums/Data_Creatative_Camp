"""Selected Mission3 model: KLUE RoBERTa-base, nine logits, fixed decision threshold."""
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForSequenceClassification
LABELS=['고열','구토','두통','복통','어지러움','열상','오심','전신쇠약','호흡곤란']
THRESHOLD=0.5
MAX_LENGTH=512
POSITIVE_WEIGHTS=[1.,1.,1.5,1.,1.,1.,2.,1.,1.]

def text_only(raw):
    """The same strip/space-join used by the source training notebook.
    Only transcript text is read; target labels and all other metadata are ignored.
    """
    return ' '.join(str(u.get('text','')).strip() for u in raw.get('utterances',[])
                    if u.get('text') is not None and str(u.get('text','')).strip())

def load_classifier(checkpoint,device):
    path=Path(checkpoint).resolve();root=path if path.is_dir() else path.parent
    weights=root/'model.pt' if path.is_dir() else path
    config=AutoConfig.from_pretrained(root,local_files_only=True)
    if config.num_labels!=len(LABELS):raise ValueError('Expected exactly nine symptom labels')
    model=AutoModelForSequenceClassification.from_config(config)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
    return model.to(device).eval(),root

@torch.inference_mode()
def probabilities(model,batch,device):
    device=torch.device(device)
    # Preserve the source inference's FP16 CUDA autocast and sigmoid precision.
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        logits=model(**{k:v.to(device) for k,v in batch.items()}).logits
    p=logits.sigmoid()
    if not torch.isfinite(p).all():raise RuntimeError('Nonfinite model probabilities')
    return p.cpu()
