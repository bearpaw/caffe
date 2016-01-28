classdef DB < handle
  % Window data layer train/test data parser
  % -----------------
  % USAGE
  % -----------------
  %   parser = window_data.DB('path_to_window_data_file', quitemode=0 or 1);
  %   db = parser.get_db();
  %   parser.show_window();
  %
  % Data structure
  % -----------------
  %     db =
  %         dbpath: 'path_to_window_data_file'
  %         imgcnt: 14218
  %         wincnt: 1112436
  %         images: [1x14218 struct]
  % -----------------
  %     db.images(i) =
  %             path: 'path_to_image.jpg'
  %         channels: 3
  %           height: 160
  %            width: 70
  %         nwindows: 26
  %          classid: [26x1 int32]
  %          overlap: [26x1 double]
  %               x1: [26x1 int32]
  %               y1: [26x1 int32]
  %               x2: [26x1 int32]
  %               y2: [26x1 int32]
  % -----------------
  % platero.yang (at) gmail.com, Feb 12, 2015
  % -----------------
  
  properties (Access = private)
    db % window dataset
  end
  
  methods
    %----------------------------------------------------------------------
    % Constructor
    %----------------------------------------------------------------------
    function this = DB(filename, quietmode, fgth, bgth)
      this.db.dbpath = filename;
      this.db.imgcnt = 0;
      this.db.wincnt = 0;
      this.db.labelhist = containers.Map({0}, [0] );
      
      if ~exist('fgth', 'var')
        fgth = 0.5;
      end
      
      if ~exist('bgth', 'var')
        bgth = 0.5;
      end
      
      assert(fgth >= bgth, 'fg threshold < bg threshold');
      
      dbfile = fopen(this.db.dbpath, 'r');
      if dbfile == -1
        fprintf('Cannot open db: %s\n', this.db.dbpath);
      end
      
      while ~feof(dbfile)
        newline = fgetl(dbfile);
        this.db.imgcnt = this.db.imgcnt + 1;
        this.db.images(this.db.imgcnt).path = fgetl(dbfile);
        this.db.images(this.db.imgcnt).channels = str2num(fgetl(dbfile));
        this.db.images(this.db.imgcnt).width = str2num(fgetl(dbfile));
        this.db.images(this.db.imgcnt).height = str2num(fgetl(dbfile));
        this.db.images(this.db.imgcnt).nwindows = str2num(fgetl(dbfile));
        
        if ~quietmode
          fprintf('%.8d %s | %d\n', this.db.imgcnt, this.db.images(this.db.imgcnt).path, this.db.images(this.db.imgcnt).nwindows)
        end
        this.db.wincnt = this.db.wincnt + this.db.images(this.db.imgcnt).nwindows;
        
        if this.db.images(this.db.imgcnt).nwindows > 0
          % read windows
          w = textscan(dbfile, '%d %f %d %d %d %d', this.db.images(this.db.imgcnt).nwindows);
          this.db.images(this.db.imgcnt).classid = w{1};
          this.db.images(this.db.imgcnt).overlap = w{2};
          this.db.images(this.db.imgcnt).x1 = w{3};
          this.db.images(this.db.imgcnt).y1 = w{4};
          this.db.images(this.db.imgcnt).x2 = w{5};
          this.db.images(this.db.imgcnt).y2 = w{6};
          
          %           for w = 1:this.db.images(this.db.imgcnt).nwindows
          %             ov = this.db.images(this.db.imgcnt).overlap(w);
          %             cls = this.db.images(this.db.imgcnt).classid(w);
          %             if ov >= fgth
          %               if ~isKey(this.db.labelhist, cls)
          %                 this.db.labelhist(cls) = 1;
          %               end
          %               this.db.labelhist(cls) = this.db.labelhist(cls) + 1;
          %             elseif ov <= bgth
          %               this.db.labelhist(0) = this.db.labelhist(0) + 1;
          %             end
          %           end
          
          %           iminfo = this.db.images(this.db.imgcnt);
          %           width = this.db.images(this.db.imgcnt).width;
          %           height = this.db.images(this.db.imgcnt).height;
          %           xmin = min(this.db.images(this.db.imgcnt).x1);
          %           ymin = min(this.db.images(this.db.imgcnt).y1);
          %           xmax = max(this.db.images(this.db.imgcnt).x2);
          %           ymax = max(this.db.images(this.db.imgcnt).y2);
          %           if xmin <=0 || ymin <=0 || ...
          %               xmax > width || ...
          %               ymax > height
          %             fprintf('%d\n', this.db.imgcnt);
          %             this.db.images(this.db.imgcnt).path
          %             im = imread(this.db.images(this.db.imgcnt).path);
          %             clf; imagesc(im); axis image; hold on;
          %             for w = 1:this.db.images(this.db.imgcnt).nwindows
          %               x1 = this.db.images(this.db.imgcnt).x1(w);
          %               x2 = this.db.images(this.db.imgcnt).x2(w);
          %               y1 = this.db.images(this.db.imgcnt).y1(w);
          %               y2 = this.db.images(this.db.imgcnt).y2(w);
          %               ov = this.db.images(this.db.imgcnt).overlap(w);
          %
          % %               if x1 <=0 || y1 <= 0 || x2 > width || y2 > height
          %                 plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'color', 'red'); hold on;
          %                 pause;
          % %               end
          %             end
          %             pause;
          %           end
          % fix flush
          fgetl(dbfile);
        end
      end
      fclose(dbfile);
    end
    
    %----------------------------------------------------------------------
    % get window_data dataset
    %----------------------------------------------------------------------
    function db = get_db(this)
      db = this.db;
    end
    
    %----------------------------------------------------------------------
    % get positive window number
    %----------------------------------------------------------------------
    function cnt = pos_num(this, thresh)
      cnt = 0;
      for i = 1:this.db.imgcnt
        
        for w = 1:this.db.images(i).nwindows
          ov = this.db.images(i).overlap(w);
          if ov >= thresh
            cnt = cnt + 1;
          end
        end
      end
    end
    %----------------------------------------------------------------------
    % get negative window number
    %----------------------------------------------------------------------
    function cnt = neg_num(this, thresh)
      cnt = 0;
      for i = 1:this.db.imgcnt
        
        for w = 1:this.db.images(i).nwindows
          ov = this.db.images(i).overlap(w);
          if ov <= thresh
            cnt = cnt + 1;
          end
        end
      end
    end
    
    %----------------------------------------------------------------------
    % Get windows
    %----------------------------------------------------------------------
    function [patches, labels] = get_window(this, index, thresh, warpx, warpy)
      im = imread(this.db.images(index).path);
      [imh, imw, imc] = size(im);
      
      patches = zeros(warpy, warpx, imc, this.db.images(index).nwindows, 'uint8');
      labels  = zeros(1, this.db.images(index).nwindows);
      
      for w = 1:this.db.images(index).nwindows
        % ---- get patch
        x1 = this.db.images(index).x1(w);
        x2 = this.db.images(index).x2(w);
        y1 = this.db.images(index).y1(w);
        y2 = this.db.images(index).y2(w);
        patches(:,:,:,w) = imresize(im(y1:y2, x1:x2, :), [warpy, warpx]);
        % ---- get label
        ov = this.db.images(index).overlap(w);
        assert(x1>0 && y1>0 && x2 < imw && y2 < imh, 'window out of boundary');
        if ov >= thresh
          labels(w) = this.db.images(index).classid(w);
        else
          labels(w) = 0;
        end
      end
    end
    %----------------------------------------------------------------------
    % visulize windows
    %----------------------------------------------------------------------
    function show_window(this, num, thresh, show_sample)
      if nargin < 2
        num = this.db.imgcnt;
      end
      if nargin < 3
        thresh = 0;
      end
      if nargin < 4
        show_sample = 'all';
      end
      for i = 1:num
        [~, imname, ext] = fileparts(this.db.images(i).path);
        im = imread(this.db.images(i).path);
        figure; title([imname, ext]);
        imshow(im); hold on;
        wincolors = colormap(jet(this.db.images(i).nwindows));
        for w = 1:this.db.images(i).nwindows
          x1 = this.db.images(i).x1(w);
          x2 = this.db.images(i).x2(w);
          y1 = this.db.images(i).y1(w);
          y2 = this.db.images(i).y2(w);
          ov = this.db.images(i).overlap(w);
          switch show_sample
            case 'all'
              plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'color', wincolors(w, :)); hold on;
            case 'pos'
              if ov >= thresh
                plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'color', wincolors(w, :)); hold on;
              end
            case 'neg'
              if ov <= thresh
                plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'color', wincolors(w, :)); hold on;
              end
          end
        end
        fprintf('\n%d: %s\n', i, this.db.images(i).path);
        disp('Press any key to continue. Ctrl + c to exit');
        pause; close;
      end
    end
    %----------------------------------------------------------------------
    % show windows by ID
    %----------------------------------------------------------------------
    function show_window_by_class(this, num, thresh, cls)
      if nargin < 2
        num = this.db.imgcnt;
      end
      if nargin < 3
        thresh = 0;
      end
      if nargin < 4
        show_sample = 'all';
      end
      cnt = 1;
      for i = 1:this.db.imgcnt
        [~, imname, ext] = fileparts(this.db.images(i).path);
        idx = find(this.db.images(i).classid == cls);
        if isempty(idx)
          continue;
        else
          cnt = cnt + 1;
          im = imread(this.db.images(i).path);
          for curidx = 1:length(idx)
            w = idx(curidx);
            x1 = this.db.images(i).x1(w);
            x2 = this.db.images(i).x2(w);
            y1 = this.db.images(i).y1(w);
            y2 = this.db.images(i).y2(w);
            ov = this.db.images(i).overlap(w);
            
            if ov >= thresh
              patch = im(y1:y2, x1:x2,:);
              imshow(patch); pause;
            end
          end
        end
        if cnt > num
          break;
        end
      end
    end
  end
end