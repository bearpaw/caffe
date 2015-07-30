function image_mean = read_mean(image_mean_file)
% read_mean()
%   read image_mean_file (.bin)

image_mean = caffe_('read_mean', image_mean_file);

end
