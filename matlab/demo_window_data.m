% demo usage of window data layer
quietmode = 1; % whether to print information while load window data file
parser = window_data.DB('../../chen-nips14-pose-dev/cache/lsp/window_data/lsp_train.txt', quietmode);
db = parser.get_db();
parser.show_window(100); % visualize 10 images
