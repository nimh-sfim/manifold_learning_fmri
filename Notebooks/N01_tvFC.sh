set -e
# Enter scripts directory
echo "++ Entering Notebooks directory..."
cd /data/SFIMJGC_HCP7T/manifold_learning_fmri/Notebooks/

# Activate miniconda
echo "++ Activating miniconda"
source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh 

# Activate vigilance environment
echo "++ Activating rapidtide environment"
conda activate opentsne

# Run the program

python ./N06_tvFC.py -ints ${path_roits} \
                     -roi_names ${path_roinames} \
                     -win_names ${path_winnames} \
                     -outZ ${out_Z} \
                     -outR ${out_R} \
                     -outZ_normed ${out_Z_normed} \
                     -outR_normed ${out_R_normed} \
                     -wls ${wls} \
                     -wss ${wss} \
                     -tr ${tr} \
                     -null ${null}
