folderPath = './10_test_data';

files = dir(fullfile(folderPath, '*.mat'));
tic;
for idx = 1:length(files)
    dataPath = fullfile(files(idx).folder, files(idx).name);
    disp(['Loading data from: ', dataPath]);  % 显示文件路径
    data = load(dataPath);
end