% ------------------------------------------------------------------------
function label = prepare_batch(image_files,batch_size, numcluster)
% ------------------------------------------------------------------------
num_images = length(image_files);
if nargin < 2
    batch_size = num_images;
end

if nargin < 3
    numcluster = 7;
end


num_images = length(image_files);
label = zeros(1, batch_size);

for i=1:num_images
    try
        if image_files(i).part == 0 && image_files(i).cluster == 0
            label(i) = 0;
        else
            label(i) = (image_files(i).part-1)*numcluster + image_files(i).cluster;
        end
    catch
        warning('Problems with file',image_files{i});
    end
end