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

python ./N11_LE_Procrustes.py -sbj_list "${sbj_list}" \
                              -input_data ${input_data} \
                              -norm_method ${norm_method} \
                              -dist ${dist} \
                              -knn ${knn} \
                              -m ${m} \
                              -drop_xxxx ${drop_xxxx}
