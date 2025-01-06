yolo11 calibrated int8 model, using nvidia T4 due to vram limit on several nvidia GPUs
code:
model.export(
        format="engine",
        int8=True,
        data="coco.yaml"
    )