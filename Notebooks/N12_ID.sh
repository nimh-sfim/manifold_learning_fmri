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

python ./N12_ID.py -tvfc ${tvfc_path} \
                   -out_local ${out_path_local} \
                   -out_global ${out_path_global} \
                   -n_jobs ${n_jobs}
                   
