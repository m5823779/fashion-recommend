#!/bin/bash
source /opt/sa-venv/AIA_aifashion_env2/bin/activate
cd /mnt/sa-nas/Docker/jocelyn.settv-it.com/app/models
python recommend_clothing.py
