grep $1 $2 | grep step | cut -d" " -f7,9 > objvals
python getmax.py 
