grep $1 $2 | cut -d" " -f2- | tr -d "," > tmp
python numdiff.py
#python calcfracdiff.py
#grep -v 6.899999999999999467e-01 fracdiff.dat | grep -v 2.999999999999999889e-01 > fractmp
#mv fractmp fracdiff.dat
