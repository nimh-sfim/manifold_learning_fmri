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

which python 

# Run the program
if [ -z ${stability+x} ]; then
   extra_args=''
else
   extra_args=' -random_seed'
fi

# Run the program
python ./N09_UMAP.py -tvfc ${path_tvfc} \
                   -out  ${path_out}  \
                   -dist ${dist} \
                   -knn  ${knn} \
                   -m    ${m} \
                   -init ${init} \
                   -min_dist ${min_dist} \
                   -alpha ${alpha} \
                   ${extra_args}
