#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N ircot-qa-flan-t5-xxl-musqiue-00
#$ -cwd                  
#$ -l h_rt=24:00:00 
#$ -l h_vmem=80G
#$ -q gpu 
#$ -pe gpu-a100 1

module load anaconda
conda activate ircot

bash run_experiment.sh ircot_qa flan-t5-xxl musique 0.0