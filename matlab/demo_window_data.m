% demo usage of window data layer
quietmode = 1; % whether to print information while load window data file
parser = window_data.DB('/home/wyang/code/pose/chen-nips14-pose/cache/lsp_win/window_data/lsp_win_train.txt', quietmode);
fprintf('positive window number: %d\n', parser.pos_num(0.5));
fprintf('negative window number: %d\n', parser.neg_num(0.5));
db = parser.get_db();
% parser.show_window(100); % visualize 10 images
