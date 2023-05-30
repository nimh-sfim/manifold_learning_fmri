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

python ./N15_Classify.py -input_path ${input_path} \
                         -features "${features}" \
                         -clf ${clf} \
                         -pid "${pid}" \
                         -output_path ${output_path} \
                         -split_mode ${split_mode} \
                         -n_jobs ${n_jobs}