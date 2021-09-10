#$ -l tmem=10G
#$ -l h_rt=48:00:00
#$ -l gpu=true,gpu_type=gtx1080ti
#$ -S /bin/bash
#$ -j y
#$ -N M2_SL
#$ -t 1-31
#$ -wd /home/jiuzhang/Final/AL_SSL/
export PATH=/share/apps/python-3.8.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.8.5-shared/lib:$LD_LIBRARY_PATH
export PATH=/share/apps/cuda-11.0/bin:/usr/local/cuda-11.0/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/cuda-11.0/lib64:/usr/local/cuda-11.0/lib:${LD_LIBRARY_PATH}
python3 -W ignore -u /home/jiuzhang/Final/AL_SSL/SupervisedRun.py ${SGE_TASK_ID}