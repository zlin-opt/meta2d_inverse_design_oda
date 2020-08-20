maxeval=$1
epi_t=$2
initfile=$3
Job=$4
ispec=$5
beta=60
np=64

mpirun -np $np strehlopt_exec \
       -options_file config \
       -filter_radius 3 \
       -filter_sigma 5 \
       -filter_beta $beta \
       -filter_choice 0 \
       -zfixed 0 \
       -image_compress_factor 1.0 \
       -print_at 1 \
       -algouter 24 \
       -alginner 24 \
       -algmaxeval $maxeval \
       -epi_t $epi_t \
       -Job $Job \
       -initial_filename $initfile \
       -ispec $ispec \
       -Nxfar 2 \
       -reuse_ksp 1 \
       
       
       
       
       


