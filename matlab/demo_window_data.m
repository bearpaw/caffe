% demo usage of window data layer
quietmode = 1; % whether to print information while load window data file
window_file = '/home/wyang/code/pose/chen-nips14-pose/cache/flic_win/window_data/flic_win_test.txt';
window_file = '/home/wyang/code/pose/chen-nips14-pose/cache/lsp_win/window_data/lsp_win_train_pure.txt';
% window_file = '/home/wyang/code/pose/chen-nips14-pose/cache/lsp_ext/window_data/lsp_ext_val.txt';
% window_file = '/home/wyang/code/pose/chen-nips14-pose/cache/flic_plus/window_data/flic_plus_train.txt';
parser = window_data.DB(window_file, quietmode);
fprintf('positive window number: %d\n', parser.pos_num(0.5));
fprintf('negative window number: %d\n', parser.neg_num(0.5));
db = parser.get_db();
% parser.show_window(100); % visualize 10 images
% parser.show_window_by_class(100,0.5, 12000);


% root_dir  = 'tmp_imageset';
% list_file = 'list.txt';  
% parser.save_db(root_dir, list_file);
