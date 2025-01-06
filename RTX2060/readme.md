exported using RTX 2060 6GB, X model not included because vram limit

settings:

model.export(
        format="engine",
        dynamic=True,
        batch=8,
        workspace=3,
        int8=True,
        data="coco.yaml",
    )