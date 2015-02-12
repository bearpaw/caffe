function f = display_network_4D(weight, layertype)
if strcmp(layertype, 'conv') 
[ksize, ~, csize, fsize] = size(weight);
if csize == 3 % first layer
    height = ceil(sqrt(fsize));
    width = ceil(size(weight,4)/height);

    im = 0.8*ones(height*(ksize+1)-1, width*(ksize+1)-1, 3);
    k_ind = 1;
    for r = 1:height
        for c = 1:width
            if k_ind>size(weight,4)
                break;
            end
            sub_im = mat2gray(weight(:,:,1:3,k_ind));
            k_ind = k_ind+1;
            x1 = (c-1)*(ksize+1)+1;
            y1 = (r-1)*(ksize+1)+1;
            x2 = x1+ksize-1;
            y2 = y1+ksize-1;
            im(y1:y2, x1:x2, :) = sub_im;
        end
    end
else
    height = min(fsize, 64);
    width = min(csize, 64);
    im = 0.8*ones(height*(ksize+1)-1, width*(ksize+1)-1, 1);
    
    for r = 1:height
        for c = 1:width
            sub_im = mat2gray(weight(:,:,c,r));
            x1 = (c-1)*(ksize+1)+1;
            y1 = (r-1)*(ksize+1)+1;
            x2 = x1+ksize-1;
            y2 = y1+ksize-1;
            im(y1:y2, x1:x2, :) = sub_im;
        end
    end
end
im = uint8(im*255);
f = figure(1);
imshow(im);
axis off
else
end