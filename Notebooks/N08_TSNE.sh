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
if [ -z "$bh_angle" ]; then
   echo " + Running without angle parameter"
   OMP_NUM_THREADS=${n_jobs} python ./N08_TSNE.py -tvfc ${path_tvfc} \
                     -out  ${path_out}  \
                     -dist ${dist} \
                     -pp  ${pp} \
                     -m    ${m} \
                     -lr ${lr} \
                     -init ${init} \
                     -norm ${norm} \
                     -n_iter ${n_iter} \
                     -n_jobs ${n_jobs} \
                     -grad_method ${grad_method} 
else
  echo " + Running with angle parameter"
  OMP_NUM_THREADS=${n_jobs} python ./N08_TSNE.py -tvfc ${path_tvfc} \
                     -out  ${path_out}  \
                     -dist ${dist} \
                     -pp  ${pp} \
                     -m    ${m} \
                     -lr ${lr} \
                     -init ${init} \
                     -norm ${norm} \
                     -n_iter ${n_iter} \
                     -n_jobs ${n_jobs} \
                     -grad_method ${grad_method} \
                     -bh_angle ${bh_angle}
fi
