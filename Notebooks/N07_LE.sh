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
if [ -z ${stability+x} ]; then
   echo "++ INFO: Regular Run --> Will use fixed random seed"
   python ./N07_LE.py -tvfc ${path_tvfc} \
                   -out  ${path_out}  \
                   -dist ${dist} \
                   -knn  ${knn} \
                   -m    ${m}
else
   echo "++ INFO: Stability Analysis --> Will generate a separate random seed."
   python ./N07_LE.py -tvfc ${path_tvfc} \
                   -out  ${path_out}  \
                   -dist ${dist} \
                   -knn  ${knn} \
                   -m    ${m} \
                   -random_seed
fi