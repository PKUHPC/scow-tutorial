# tutorial
SCOW HPC和XSCOW HPC的教程

## 生成web页面
```bash
# 用Ubuntu 20.04.6 LTS系统, 需提前装好conda
pip install notebook beautifulsoup4
conda install pandoc
conda install -c conda-forge parallel 
sudo apt-get update
sudo apt-get install texlive-xetex google-chrome-stable pdftk

bash release.sh
```
