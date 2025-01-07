Super Fast yolo model using tensorRT, up to 3x FASTER than pytorch (.pt)

this model using int8, calibrated using coco2017 dataset

install all required library before use
```bash
pip install -r requirement.txt
```

Only use same GPU as folder name

ALL GPU Support most of nvidia gpu, but this model is 20-50% slower than int8 version

Export your own model using yoloexport.ipynb or watch this video

https://www.youtube.com/watch?v=RDWbKB-nvew on time 1:39:32

to use the model, just use it as if you're using normal yolo model, if crash make sure batch is 8 or 1.