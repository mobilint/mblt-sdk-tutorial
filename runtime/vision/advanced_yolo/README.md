```bash
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

```bash
python organize_coco.py --image_dir ./val2017.zip --annotation_dir  ./annotations_trainval2017.zip
```

```bash
python benchmark_coco.py --local_path ./yolo11m_single.mxq --infer_mode single --batch_size 8
```