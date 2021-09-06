#$ -l tmem=10G
#$ -l h_rt=24:00:00
#$ -l gpu=true,gpu_type=gtx1080ti
#$ -S /bin/bash
#$ -j y
#$ -N AL_robust
#$ -t 1-20
#$ -wd /home/jiuzhang/Final/AL_Robust/
export PATH=/share/apps/python-3.8.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.8.5-shared/lib:$LD_LIBRARY_PATH
export PATH=/share/apps/cuda-11.0/bin:/usr/local/cuda-11.0/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/cuda-11.0/lib64:/usr/local/cuda-11.0/lib:${LD_LIBRARY_PATH}
python3 -W ignore -u /home/jiuzhang/Final/AL_Robust/RobustnessRun.py ${SGE_TASK_ID}
