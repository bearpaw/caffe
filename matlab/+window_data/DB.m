classdef DB < handle
% Window data layer train/test data parser
% -----------------
% USAGE
% -----------------
%   parser = window_data.DB('path_to_window_data_file');
%   db = parser.get_db();
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
    %-------------------------
    % Constructor
    function this = DB(filename)
        this.db.dbpath = filename;
        this.db.imgcnt = 0;
        this.db.wincnt = 0;
        
        dbfile = fopen(this.db.dbpath, 'r');
        if dbfile == -1
            fprintf('Cannot open db: %s\n', this.db.dbpath);
        end

        while ~feof(dbfile)
            newline = fgetl(dbfile);
            this.db.imgcnt = this.db.imgcnt + 1;
            this.db.images(this.db.imgcnt).path = fgetl(dbfile);
            this.db.images(this.db.imgcnt).channels = str2num(fgetl(dbfile));
            this.db.images(this.db.imgcnt).height = str2num(fgetl(dbfile));
            this.db.images(this.db.imgcnt).width = str2num(fgetl(dbfile));
            this.db.images(this.db.imgcnt).nwindows = str2num(fgetl(dbfile));
            
            fprintf('%.8d %s | %d\n', this.db.imgcnt, this.db.images(this.db.imgcnt).path, this.db.images(this.db.imgcnt).nwindows)
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
                % fix flush
                fgetl(dbfile);
            end
        end
        fclose(dbfile);
    end
    
    %-------------------------
    % get window_data dataset    
    function db = get_db(this)
        db = this.db;
    end
    
end
end