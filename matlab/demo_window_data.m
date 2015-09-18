% demo usage of window data layer
quietmode = 1; % whether to print information while load window data file
% parser = window_data.DB('/home/wyang/code/pose/chen-nips14-pose/cache/lsp_win/window_data/lsp_win_test.txt', quietmode);
parser = window_data.DB('/home/wyang/code/pose/chen-nips14-pose/cache/lsp_win_72/window_data/lsp_win_72_test.txt', quietmode);

db = parser.get_db();
parser.show_window(100); % visualize 10 images
