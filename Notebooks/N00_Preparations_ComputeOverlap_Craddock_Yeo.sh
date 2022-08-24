#!/bash/bin
set -e

cd /data/SFIMJGC_HCP7T/manifold_learning_fmri/Data/PNAS2015/SBJ06

3dcalc -overwrite \
       -a SBJ06.Yeo2011_7Networks_01_Visual+orig. \
       -b SBJ06.Yeo2011_7Networks_02_SomatoMotor+orig. \
       -c SBJ06.Yeo2011_7Networks_03_DorsalAttention+orig. \
       -d SBJ06.Yeo2011_7Networks_04_VentralAttention+orig. \
       -e SBJ06.Yeo2011_7Networks_05_Limbic+orig. \
       -f SBJ06.Yeo2011_7Networks_06_Control+orig. \
       -g SBJ06.Yeo2011_7Networks_07_DMN+orig. \
       -expr 'a+b+c+d+e+f+g' \
       -prefix SBJ06.Yeo2011_7Networks_All
3dmask_tool -overwrite \
            -input SBJ06.Yeo2011_7Networks_All+orig. \
            -fill_dirs xy -fill_holes \
            -prefix SBJ06.Yeo2011_7Networks_All.filled 
3dcalc -overwrite \
       -a SBJ06.Yeo2011_7Networks_All.filled+orig. \
       -b SBJ06.Yeo2011_7Networks_All+orig. \
       -expr 'a-b' -prefix SBJ06.Yeo2011_7Networks_08_Subcortical

rm SBJ06.Yeo2011_7Networks_All+orig.* SBJ06.Yeo2011_7Networks_All.filled+orig.*


echo "ROI_Number,Num_Voxels,01_Visual,02_SomatoMotor,03_DorsalAttention,04_VentralAttention,05_Limbic,06_Control,07_DMN,08_Subcortical" > SBJ06.Craddock2Yeo_Overlap.csv
for r in `seq 1 1 157`
do
    line=`echo "${r}"`
    3dcalc -overwrite -a SBJ06_Craddock_0200+orig.HEAD -expr "equals(a,${r})" -prefix rm.ROI${r}.nii
    # Compute Number of Voxels in the ROI
    NV=`3dROIstats -nomeanout -quiet -nzvoxels -mask rm.ROI${r}.nii rm.ROI${r}.nii  | awk '{print $1}'`
    if [[ -z "${NV}" ]]; then NV=0; fi
    line=`echo "${line},${NV}"`
    # Compute number of voxel that overlap per network
    for nw in 01_Visual 02_SomatoMotor 03_DorsalAttention 04_VentralAttention 05_Limbic 06_Control 07_DMN 08_Subcortical
    do
        3dcalc -overwrite -a rm.ROI${r}.nii -b SBJ06.Yeo2011_7Networks_${nw}+orig -expr 'step(a)*step(b)' -prefix rm.${r}.${nw}.nii
        NV=`3dROIstats -nomeanout -quiet -nzvoxels -mask rm.${r}.${nw}.nii rm.${r}.${nw}.nii  | awk '{print $1}'`
        if [[ -z "${NV}" ]]; then NV=0; fi
        rm rm.${r}.${nw}.nii
        line=`echo "${line},${NV}"`
    done
    rm rm.ROI${r}.nii
    echo "$line" >> SBJ06.Craddock2Yeo_Overlap.csv
done
