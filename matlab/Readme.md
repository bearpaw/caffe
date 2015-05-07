## Visualize training log
Call the function `visualize_log` by
```matlab
visualize_log 'log_file_path'
```

## Window Data Parser
Simple parser of window data file
### USAGE
```matlab
  parser = window_data.DB('path_to_window_data_file');
  db = parser.get_db();
```

### Data structure
```matlab
    db = 
        dbpath: 'path_to_window_data_file'
        imgcnt: 14218
        wincnt: 1112436
        images: [1x14218 struct]


    db.images(i) =
            path: 'path_to_image.jpg'
        channels: 3
          height: 160
           width: 70
        nwindows: 26
         classid: [26x1 int32]
         overlap: [26x1 double]
              x1: [26x1 int32]
              y1: [26x1 int32]
              x2: [26x1 int32]
              y2: [26x1 int32]
```
