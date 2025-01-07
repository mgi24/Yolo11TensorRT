Super Fast yolo model using tensorRT, up to 3x FASTER than pytorch (.pt)

this model using int8, calibrated using coco2017 dataset

install all required library before use
```bash
pip install -r requirement.txt
```

YOU CANNOT USE ANY OF THE MODEL IN THIS REPO!, since every model only can be used
by the same GPU and same tensorRT version they're exported from

To export it yourself please watch
https://www.youtube.com/watch?v=RDWbKB-nvew on time 1:39:32
or open yoloexport.ipynb

to use the model, just use it as if you're using normal yolo model, if crash make sure batch is 8 or 1.