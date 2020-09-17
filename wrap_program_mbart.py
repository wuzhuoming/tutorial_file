import os
import nni
import time

params = {
  'dropout':0.0,
  'label_smooth': 0.1,
  'lr':0.00001,
  'lr_scheduler':"inverse_sqrt",
  'warmup_update':2500,
  'optimizer':"adam",
  'inter_op_parallelism_threads':1,
  'intra_op_parallelism_threads':2,
  'benchmark':0,
  'allow_tf32':0,
} 

tuned_params = nni.get_next_parameter() 
params.update(tuned_params) 
t_id = nni.get_trial_id()


langs="ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN"
path_2_data="/research/d3/zmwu/model/mbart/postprocessed/en-ja"
pretrained_model="/research/d3/zmwu/model/mbart/mbart.cc25"
user_dir="/research/d3/zmwu/model/mbart/mbart"
save_dir="/research/d3/zmwu/model/mbart/checkpoints/" + t_id
lang_pairs = "en-ja"
src = "en_XX"
tgt = "ja_XX"


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

s_time = time.time()
os.system("fairseq-train %s \
  --user-dir %s \
  --save-dir %s \
  --finetune-from-model %s \
  --encoder-normalize-before --decoder-normalize-before \
  --arch transformer --layernorm-embedding \
  --task translation_multi_simple_epoch_nni \
  --sampling-method \"temperature\" \
  --sampling-temperature 1.5 \
  --encoder-langtok \"src\" \
  --decoder-langtok \
  --langs %s \
  --lang-pairs %s \
  -s %s -t %s\
  --criterion label_smoothed_cross_entropy  \
   --min-lr -1  --max-update 40000 \
  --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 10 --max-epoch 50 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 1 \
	--dropout %f --label-smoothing %f --lr %f  --lr-scheduler %s --warmup-updates %d --optimizer %s \
	--inter %d --intra %d --benchmark %d --allow_tf32 %d"
	%(path_2_data,
	  user_dir,
	  save_dir,
    pretrained_model,
    langs,
    lang_pairs,
    src,
    tgt,
    params['dropout'],params['label_smooth'],params['lr'],params['lr_scheduler'],params['warmup_update'],params['optimizer'],
    int(params['inter_op_parallelism_threads']),int(params['intra_op_parallelism_threads']),int(params['benchmark']),int(params['allow_tf32'])))
e_time = time.time()

spent_time = (e_time - s_time) / 3600.0
print(spent_time)


ckpt_path = save_dir+"/checkpoint_last.pt"
gen_subset = "valid"

spm = "/research/d3/zmwu/model/mbart/mbart.cc25/sentence.bpe.model"



os.system("fairseq-generate \
--path=%s %s \
--user-dir %s \
--task translation_multi_simple_epoch_nni \
--encoder-langtok 'src' --decoder-langtok \
--gen-subset %s \
-s %s -t %s \
--langs %s --lang-pairs %s \
--bpe 'sentencepiece' --sentencepiece-model %s \
--scoring 'sacrebleu' --fp16 --max-sentences 128 \
--results-path %s "
%(ckpt_path,
  path_2_data,
  user_dir,
  gen_subset,
  src,tgt,
  langs,lang_pairs,
  spm,
  save_dir))

target_file = "path_2_target_file"
result_file = "path_2_result_file"

bsf = os.popen("cat %s | grep -P \"^D\" | cut -f 3- | sacrebleu -b %s"%(result_file,target_file))
bs = float(bsf.readline())
print(bs)


report_dict = {'runtime':spent_time,'default':bs,'maximize':['default']}
nni.report_final_result(report_dict)
