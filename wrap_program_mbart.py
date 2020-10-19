import os
import nni
import time
import signal
import subprocess
import shlex
import psutil
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

def kill_process_and_children(proc_pid):
  process = psutil.Process(proc_pid)
  for proc in process.children(recursive=True):
      proc.kill()
  process.kill()

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


path_2_data="/research/d3/zmwu/model/mbart_company_version/post_process/en-zh_100/"
lang_pairs="en_XX-zh_CN,zh_CN-en_XX"
lang_list="/research/d3/zmwu/model/mbart_company_version/lang_list"
pretrained_model="/research/d3/zmwu/model/mbart_company_version/mbart.cc25/model.pt"
user_dir="/research/d3/zmwu/model/mbart_company_version/mbart/"
save_dir="/research/d3/zmwu/model/mbart_company_version/ckpt/" + t_id

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_cmd = "fairseq-train %s --user-dir %s --save-dir %s --finetune-from-model %s --encoder-normalize-before --decoder-normalize-before --arch mbart_large --layernorm-embedding --task translation_multi_simple_epoch_nni --sampling-method \"temperature\" --sampling-temperature 1.5 --encoder-langtok \"src\" --decoder-langtok --lang-dict \"%s\" --lang-pairs \"%s\" --criterion label_smoothed_cross_entropy --min-lr -1  --max-update 20000  --empty-cache-freq 4 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 2048 --update-freq 4 --fp16 --save-interval-updates 20000 --keep-interval-updates 2 --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 1 --dropout %f --label-smoothing %f --lr %f  --lr-scheduler %s --warmup-updates %d --optimizer %s --inter %d --intra %d --benchmark %d --allow_tf32 %d"%(path_2_data,user_dir,save_dir,pretrained_model,lang_list,lang_pairs,params['dropout'],params['label_smooth'],params['lr'],params['lr_scheduler'],params['warmup_update'],params['optimizer'],int(params['inter_op_parallelism_threads']),int(params['intra_op_parallelism_threads']),int(params['benchmark']),int(params['allow_tf32']))


s_time = time.time()

train_process = subprocess.Popen(shlex.split(train_cmd),shell=False)
train_pid = train_process.pid
logging.info("train process start,process ID is %d" % train_pid)
train_process.wait()
logging.info('train process finish, check if train process close properly...')
if psutil.pid_exists(train_pid):
  logging.info("trian process still exists, kill it.")
  kill_process_and_children(train_pid)
else:
  logging.info("train process finish and exit normally.")

e_time = time.time()
spent_time = (e_time - s_time) / 3600.0
logging.info("time spenting on training: %fh" % spent_time)


ckpt_path = save_dir+"/checkpoint_last.pt"
gen_subset = "test"
spm = "/research/d3/zmwu/model/mbart/mbart.cc25/sentence.bpe.model"

generate_cmd = "fairseq-generate --path=%s %s --user-dir %s --task translation_multi_simple_epoch_nni --encoder-langtok 'src' --decoder-langtok --gen-subset %s -s %s -t %s --lang-dict %s --lang-pairs %s --bpe 'sentencepiece' --empty-cache-freq 1 --sentencepiece-model %s --scoring 'sacrebleu' --fp16 --max-sentences 128 --results-path %s"%(ckpt_path,path_2_data,user_dir,gen_subset,"en_XX","zh_CN",lang_list,lang_pairs,spm,save_dir)


generate_process = subprocess.Popen(shlex.split(generate_cmd),stdout=outf,shell=False)
generate_pid = generate_process.pid
logging.info("generate process start,process ID is %d" % generate_pid) 
generate_process.wait()
logging.info('generate process finish, check if generate process close properly...')
if psutil.pid_exists(generate_pid):
  logging.info("generate process still exists, kill it.")
  kill_process_and_children(generate_pid)
else:
  logging.info("generate process finish and exit normally.")


target_file = "/research/d3/zmwu/model/mbart_company_version/ref-legal-en-zh"
result_file = "%s/generate-test.txt"%(save_dir)


os.system("cat %s | grep -P \"^T\" | cut -f 2- > %s"%(result_file, target_file))
bsf = os.popen("cat %s | grep -P \"^D\" | cut -f 3- | sacrebleu -b --tok zh %s"%(result_file,target_file))
bs = float(bsf.readline())
logging.info("bleu score: %d" % bs)


report_dict = {'runtime':spent_time,'default':bs,'maximize':['default']}
nni.report_final_result(report_dict)