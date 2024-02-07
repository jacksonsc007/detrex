from detrex.config import get_config
from .models.conditional_detr_r50 import model
from configs.common.voc_schedule import default_voc_scheduler

dataloader = get_config("common/data/voc_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

batch_size = 4
total_imgs = 16651
num_epochs = 12
if num_epochs == 12:
    lr_multiplier = default_voc_scheduler(12, 11, 0, batch_size)
elif num_epochs == 50:
    lr_multiplier = default_voc_scheduler(50, 40, 0, batch_size)
else:
    raise ValueError("num_epoch does not support for now") 


setting_code = f"bs{batch_size}_epoch{num_epochs}"

# ============= dataloader config =========================

dataloader.train.total_batch_size = batch_size
dataloader.train.num_workers = 8

# dump the testing results into output_dir for visualization
# NOTE that VOC evaluator don't need output_dir
dataloader.evaluator.output_dir = train.output_dir

dataset_code = "voc"

# ============ modify optimizer config ===================
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

optim_code = f"lr{optimizer.lr}"

# ============ model config ==============================
model.device = train.device

# for VOC dataset
model.num_classes = 20
model.criterion.num_classes = 20
model.transformer.encoder.num_layers=2 
model.transformer.decoder.num_layers=2 

model_code = f"conditional_detr_enc{model.transformer.encoder.num_layers}_dec{model.transformer.decoder.num_layers}"

# ============= train config ==============================

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.max_iter = int( num_epochs * total_imgs / batch_size)
# run evaluation every 5000 iters
train.eval_period = 5000
# log training infomation every 20 iters
train.log_period = 20
# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"

# wandb log
train.wandb.enabled = True
train.wandb.params.name = "-".join(["bug_fixing", model_code, dataset_code, setting_code, optim_code, ])
train.output_dir = "./output/" + "${train.wandb.params.name}"
#print(train.output_dir)
#exit()





