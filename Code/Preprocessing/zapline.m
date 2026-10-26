% ============================================================
% zapline.m
%
% Methods 3.3 (EEG preprocessing): removes the narrowband 40 Hz
% stimulation-frequency artifact from the RELAX-cleaned 40 Hz (Cond3)
% recordings using Zapline-plus (noisefreqs = 40). All later 40 Hz source
% analyses use these ZapLine-cleaned .set files; other conditions use the
% standard RELAX-cleaned files.
%
% Set the two addpath lines below to the local EEGLAB and Zapline-plus
% installations before running.
% ============================================================

addpath(genpath('C:\path\to\eeglab'))
addpath(genpath('C:\path\to\zapline-plus'))




%% Paths
projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
inputFolder  = fullfile(projectRoot, 'Data', 'raw_cleaned_set_files');
outputFolder = fullfile(projectRoot, 'outputs', 'zapline_40');

%% Create output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

%% Get only Cond3 .set files
files = dir(fullfile(inputFolder, '*Cond3*.set'));

%% Loop through files
for i = 1:length(files)

    fprintf('\nProcessing file %d/%d\n', i, length(files));
    fprintf('File: %s\n', files(i).name);

    %% Load EEG dataset
    EEG = pop_loadset( ...
        'filename', files(i).name, ...
        'filepath', inputFolder);

    %% Convert data to double
    data = double(EEG.data);

    %% Run Zapline-plus at 40 Hz
    cleanData = clean_data_with_zapline_plus( ...
        data, ...
        EEG.srate, ...
        'noisefreqs', 40);

    %% Put cleaned data back into EEG structure
    EEG.data = cleanData;

    %% Save cleaned dataset
    pop_saveset( ...
        EEG, ...
        'filename', files(i).name, ...
        'filepath', outputFolder);

    fprintf('Saved cleaned file.\n');

end

fprintf('\nAll Cond3 files processed.\n');
