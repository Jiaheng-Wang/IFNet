%%
%parse BCIC-IV-2A data 
%training set load
clear
close all
dataset_dir = '~/data/';
labels_dir = '~/labels/';
dataset_name='A01T';
save_dir = '~/';
eeglab;
EEG = pop_biosig([dataset_dir dataset_name '.gdf']);
EEG.setname = dataset_name;
EEG = eeg_checkset( EEG );
EEG = pop_select( EEG, 'nochannel',{'EOG-left','EOG-central','EOG-right'});
EEG = eeg_checkset( EEG );
EEG = pop_epoch( EEG, {  'class1, Left hand	- cue onset (BCI experiment)'  'class2, Right hand	- cue onset (BCI experiment)' 'class3, Foot, towards Right - cue onset (BCI experiment)' 'class4, Tongue		- cue onset (BCI experiment)'},[0  4], 'newname', 'A01T epochs', 'epochinfo', 'yes');
EEG = eeg_checkset( EEG );
load([labels_dir '\' dataset_name '.mat'])
indexes =(classlabel == 1)+(classlabel == 2)+(classlabel == 3)+(classlabel == 4);
indexes = indexes == 1;
EEG_data=EEG.data(:,:,indexes);
labels = classlabel(indexes);
file = strcat([save_dir dataset_name(1:end-1) '\training.mat']);
mkdir(strcat([save_dir dataset_name(1:end-1)]));
save(file, 'EEG_data', 'labels');

%%
%parse BCIC-IV-2A data 
%evaluation set load
clear
close all
dataset_dir = '';
labels_dir = '';
dataset_name='A01E';
save_dir = '';
eeglab
EEG = pop_biosig([dataset_dir dataset_name '.gdf']);
EEG.setname = dataset_name;
EEG = eeg_checkset( EEG );
EEG = pop_select( EEG, 'nochannel',{'EOG-left','EOG-central','EOG-right'});
EEG = eeg_checkset( EEG );
EEG = pop_epoch( EEG, {  'cue unknown/undefined (used for BCI competition) '}, [0  4], 'newname', 'A01T epochs', 'epochinfo', 'yes');
EEG = eeg_checkset( EEG );
load([labels_dir '\' dataset_name '.mat'])
indexes =(classlabel == 1)+(classlabel == 2)+(classlabel == 3)+(classlabel == 4);
indexes = indexes == 1;
EEG_data = EEG.data(:,:,indexes);
labels = classlabel(indexes);
file = strcat([save_dir dataset_name(1:end-1) '\evaluation.mat']);
mkdir(strcat([save_dir dataset_name(1:end-1)]));
save(file, 'EEG_data', 'labels');