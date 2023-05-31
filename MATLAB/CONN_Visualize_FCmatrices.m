close all
clear all
% TO GENERATE A COORDINATE FILE ASSOCIATED WITH THE ATLAS WE USE IN THIS
% PROJECT.
% cd /data/SFIMJGC/PRJ_CognitiveStateDetection01/PrcsData/SALL
% 3dCM -all_rois SALL.Craddock_T2Level_0200.MNI+tlrc. | grep -v '#' | tr -s ' ' ',' > SALL.Craddock_T2Level_0200.MNI.coordinates.1D
% NOTE: For convenience this atlas is also available at: 
% /data/SFIMJGC_HCP7T/manifold_learning_fmri/Outputs/Corner_FC_4CONN

% Start Parallel Pool (for efficiency)
% Make CONN and SPM accessible
addpath('/opt/matlab/conn')
addpath('/opt/matlab/spm12')

% Load One Connectivity Matrix
fc = load('~/Downloads/conn_files/manifold/Fig01_Corner08_FC.txt');
fc = fc .* (abs(fc)>.3);

% Load ROI Labsls
roi_coords_arr=load('~/Downloads/conn_files/manifold/SALL.Craddock_T2Level_0200.MNI.coordinates.1D');
roi_coords = {};

for i =1:157
    roi_coords{i} = roi_coords_arr(i,:);
end
roi_coords_arr(:,2) = -roi_coords_arr(:,2) % Becuase 3dCM did not do an excellent job here.
global CONN_gui; CONN_gui.usehighres=true;
cd = conn_mesh_display('','','',roi_coords_arr,fc);