'''
PS C:\Tools\Quantum_Classification> py -3.13 Benchmark_Clean_Eval.py

--- Initializing Benchmark Pipeline on cuda ---
Loading OOD and Train Data...

=============================================
🚀 Training ResNet18
=============================================
  Epoch 1/15 | Val Acc: 0.9965
  Epoch 2/15 | Val Acc: 0.9895
  Epoch 3/15 | Val Acc: 0.9955
  Epoch 4/15 | Val Acc: 0.9930
  Epoch 5/15 | Val Acc: 0.9920
  Epoch 6/15 | Val Acc: 0.9925
  Epoch 7/15 | Val Acc: 0.9965
  Epoch 8/15 | Val Acc: 0.9965
  Epoch 9/15 | Val Acc: 0.9995
  Epoch 10/15 | Val Acc: 1.0000
  Epoch 11/15 | Val Acc: 1.0000
  Epoch 12/15 | Val Acc: 0.9980
  Epoch 13/15 | Val Acc: 0.9970
  Epoch 14/15 | Val Acc: 0.9990
  Epoch 15/15 | Val Acc: 0.9925
✅ Model saved to saved_models\ResNet18.pth
📊 Evaluating ResNet18 on OOD DatasetNinja...
  --> Acc: 0.9724 | Prec: 0.9796 | Rec: 0.9650 | F1: 0.9723 | AUC: 0.9954 | FPS: 58.6

=============================================
🚀 Training MobileNetV3
=============================================
  Epoch 1/15 | Val Acc: 0.9905
  Epoch 2/15 | Val Acc: 0.9955
  Epoch 3/15 | Val Acc: 0.9975
  Epoch 4/15 | Val Acc: 1.0000
  Epoch 5/15 | Val Acc: 0.9955
  Epoch 6/15 | Val Acc: 1.0000
  Epoch 7/15 | Val Acc: 0.9995
  Epoch 8/15 | Val Acc: 0.9995
  Epoch 9/15 | Val Acc: 0.9915
  Epoch 10/15 | Val Acc: 0.9980
  Epoch 11/15 | Val Acc: 1.0000
  Epoch 12/15 | Val Acc: 1.0000
  Epoch 13/15 | Val Acc: 0.9995
  Epoch 14/15 | Val Acc: 0.9995
  Epoch 15/15 | Val Acc: 0.9990
✅ Model saved to saved_models\MobileNetV3.pth
📊 Evaluating MobileNetV3 on OOD DatasetNinja...
  --> Acc: 0.9879 | Prec: 0.9914 | Rec: 0.9844 | F1: 0.9879 | AUC: 0.9966 | FPS: 64.8

=============================================
🚀 Training EfficientNet-B0
=============================================
  Epoch 1/15 | Val Acc: 0.9960
  Epoch 2/15 | Val Acc: 0.9980
  Epoch 3/15 | Val Acc: 0.9995
  Epoch 4/15 | Val Acc: 1.0000
  Epoch 5/15 | Val Acc: 1.0000
  Epoch 6/15 | Val Acc: 0.9995
  Epoch 7/15 | Val Acc: 0.9995
  Epoch 8/15 | Val Acc: 0.9995
  Epoch 9/15 | Val Acc: 1.0000
  Epoch 10/15 | Val Acc: 1.0000
  Epoch 11/15 | Val Acc: 0.9995
  Epoch 12/15 | Val Acc: 1.0000
  Epoch 13/15 | Val Acc: 1.0000
  Epoch 14/15 | Val Acc: 0.9995
  Epoch 15/15 | Val Acc: 0.9975
✅ Model saved to saved_models\EfficientNet-B0.pth
📊 Evaluating EfficientNet-B0 on OOD DatasetNinja...
  --> Acc: 0.9896 | Prec: 0.9896 | Rec: 0.9896 | F1: 0.9896 | AUC: 0.9969 | FPS: 57.3

=============================================
🚀 Training Custom_Legacy_CNN
=============================================
  Epoch 1/15 | Val Acc: 0.9057
  Epoch 2/15 | Val Acc: 0.9393
  Epoch 3/15 | Val Acc: 0.9328
  Epoch 4/15 | Val Acc: 0.9669
  Epoch 5/15 | Val Acc: 0.9724
  Epoch 6/15 | Val Acc: 0.9769
  Epoch 7/15 | Val Acc: 0.9809
  Epoch 8/15 | Val Acc: 0.9814
  Epoch 9/15 | Val Acc: 0.9849
  Epoch 10/15 | Val Acc: 0.9890
  Epoch 11/15 | Val Acc: 0.9900
  Epoch 12/15 | Val Acc: 0.9865
  Epoch 13/15 | Val Acc: 0.9915
  Epoch 14/15 | Val Acc: 0.9940
  Epoch 15/15 | Val Acc: 0.9920
✅ Model saved to saved_models\Custom_Legacy_CNN.pth
📊 Evaluating Custom_Legacy_CNN on OOD DatasetNinja...
  --> Acc: 0.8799 | Prec: 0.8810 | Rec: 0.8790 | F1: 0.8800 | AUC: 0.9459 | FPS: 57.5

=============================================
🚀 Training Hybrid CNN-QNN (From Scratch)
=============================================
  Epoch 1/15 | Val Acc: 0.9644
  Epoch 2/15 | Val Acc: 0.9528
  Epoch 3/15 | Val Acc: 0.9759
  Epoch 4/15 | Val Acc: 0.9378
  Epoch 5/15 | Val Acc: 0.9784
  Epoch 6/15 | Val Acc: 0.9799
  Epoch 7/15 | Val Acc: 0.9704
  Epoch 8/15 | Val Acc: 0.9915
  Epoch 9/15 | Val Acc: 0.9920
  Epoch 10/15 | Val Acc: 0.9905
  Epoch 11/15 | Val Acc: 0.9794
  Epoch 12/15 | Val Acc: 0.9900
  Epoch 13/15 | Val Acc: 0.9839
  Epoch 14/15 | Val Acc: 0.9754
  Epoch 15/15 | Val Acc: 0.8695
✅ Model saved to saved_models\Hybrid_CNN-QNN_(From_Scratch).pth
📊 Evaluating Hybrid CNN-QNN (From Scratch) on OOD DatasetNinja...
  --> Acc: 0.9249 | Prec: 0.9264 | Rec: 0.9233 | F1: 0.9249 | AUC: 0.9675 | FPS: 55.6

=============================================
🚀 Training Custom_Legacy_QNN
=============================================
  Epoch 1/15 | Val Acc: 0.7311
  Epoch 2/15 | Val Acc: 0.8746
  Epoch 3/15 | Val Acc: 0.9363
  Epoch 4/15 | Val Acc: 0.9142
  Epoch 5/15 | Val Acc: 0.9147
  Epoch 6/15 | Val Acc: 0.9207
  Epoch 7/15 | Val Acc: 0.9237
  Epoch 8/15 | Val Acc: 0.9373
  Epoch 9/15 | Val Acc: 0.9443
  Epoch 10/15 | Val Acc: 0.9272
  Epoch 11/15 | Val Acc: 0.9313
  Epoch 12/15 | Val Acc: 0.9222
  Epoch 13/15 | Val Acc: 0.9267
  Epoch 14/15 | Val Acc: 0.9438
  Epoch 15/15 | Val Acc: 0.9488
✅ Model saved to saved_models\Custom_Legacy_QNN.pth
📊 Evaluating Custom_Legacy_QNN on OOD DatasetNinja...
  --> Acc: 0.5313 | Prec: 0.8366 | Rec: 0.0800 | F1: 0.1461 | AUC: 0.7677 | FPS: 61.3

=============================================
🚀 Training Hybrid CNN-QNN (Transfer Learning)
=============================================
  Epoch 1/15 | Val Acc: 0.9644
  Epoch 2/15 | Val Acc: 0.9739
  Epoch 3/15 | Val Acc: 0.9769
  Epoch 4/15 | Val Acc: 0.9799
  Epoch 5/15 | Val Acc: 0.9860
  Epoch 6/15 | Val Acc: 0.9829
  Epoch 7/15 | Val Acc: 0.9860
  Epoch 8/15 | Val Acc: 0.9895
  Epoch 9/15 | Val Acc: 0.9839
  Epoch 10/15 | Val Acc: 0.9875
  Epoch 11/15 | Val Acc: 0.9880
  Epoch 12/15 | Val Acc: 0.9870
  Epoch 13/15 | Val Acc: 0.9895
  Epoch 14/15 | Val Acc: 0.9945
  Epoch 15/15 | Val Acc: 0.9910
✅ Model saved to saved_models\Hybrid_CNN-QNN_(Transfer_Learning).pth
📊 Evaluating Hybrid CNN-QNN (Transfer Learning) on OOD DatasetNinja...
  --> Acc: 0.9239 | Prec: 0.9022 | Rec: 0.9512 | F1: 0.9261 | AUC: 0.9784 | FPS: 56.1

=============================================
🚀 Training YOLOv8n-cls
=============================================
New https://pypi.org/project/ultralytics/8.4.71 available  Update with 'pip install -U ultralytics'
Ultralytics 8.4.64  Python-3.13.14 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 8188MiB)
engine\trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=15, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=128, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8n-cls.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=train-11, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=C:\Tools\Quantum_Classification\runs\classify\train-11, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=classify, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=False, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=0, workspace=None
train: C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit\train... found 2393 images in 2 classes
val: None...
test: C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit\test... found 1993 images in 2 classes
Overriding model.yaml nc=1000 with nc=2

                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    332802  ultralytics.nn.modules.head.Classify         [256, 2]
YOLOv8n-cls summary: 56 layers, 1,440,850 parameters, 1,440,850 gradients, 3.4 GFLOPs
Transferred 156/158 items from pretrained weights
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed
train: Fast image access  (ping: 0.00.0 ms, read: 1941.6624.6 MB/s, size: 201.2 KB)
train: Scanning C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit\train... 2393 images, 0 corrupt: 100% ━━━━━━━━━━━━ 2393/2393 116.7Mit/s 0.0s
val: Fast image access  (ping: 0.00.0 ms, read: 2325.6559.8 MB/s, size: 201.2 KB)
val: Scanning C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit\test... 921 images, 0 corrupt: 46% ━━━val: Scanning C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit\test... 1838 images, 0 corrupt: 92% ━━val: Scanning C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit\test... 1993 images, 0 corrupt: 100% ━━━━━━━━━━━━ 1993/1993 9.2Kit/s 0.2s
val: New cache created: C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit\test.cache
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically...
optimizer: AdamW(lr=0.001667, momentum=0.9) with parameter groups 26 weight(decay=0.0), 27 weight(decay=0.0005), 27 bias(decay=0.0)
Image sizes 128 train, 128 val
Using 0 dataloader workers
Logging results to C:\Tools\Quantum_Classification\runs\classify\train-11
Starting training for 15 epochs...

      Epoch    GPU_mem       loss  Instances       Size
       1/15      2.58G     0.4134         25        128: 100% ━━━━━━━━━━━━ 75/75 5.3it/s 14.1s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.7it/s 6.9s
                   all      0.985          1

      Epoch    GPU_mem       loss  Instances       Size
       2/15      2.59G     0.1259         25        128: 100% ━━━━━━━━━━━━ 75/75 5.5it/s 13.6s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.6it/s 7.0s
                   all       0.99          1

      Epoch    GPU_mem       loss  Instances       Size
       3/15       2.6G     0.1236         25        128: 100% ━━━━━━━━━━━━ 75/75 5.6it/s 13.5s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.6it/s 7.0s
                   all      0.995          1

      Epoch    GPU_mem       loss  Instances       Size
       4/15      2.61G    0.07513         25        128: 100% ━━━━━━━━━━━━ 75/75 5.5it/s 13.6s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.6it/s 6.9s
                   all      0.998          1

      Epoch    GPU_mem       loss  Instances       Size
       5/15      2.62G    0.06253         25        128: 100% ━━━━━━━━━━━━ 75/75 5.8it/s 13.0s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.5it/s 7.1s
                   all      0.999          1

      Epoch    GPU_mem       loss  Instances       Size
       6/15      2.62G    0.07528         25        128: 100% ━━━━━━━━━━━━ 75/75 5.4it/s 14.0s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.6it/s 6.9s
                   all          1          1

      Epoch    GPU_mem       loss  Instances       Size
       7/15      2.63G    0.03894         25        128: 100% ━━━━━━━━━━━━ 75/75 5.7it/s 13.2s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.6it/s 7.0s
                   all          1          1

      Epoch    GPU_mem       loss  Instances       Size
       8/15      2.64G    0.04946         25        128: 100% ━━━━━━━━━━━━ 75/75 5.7it/s 13.1s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.6it/s 6.9s
                   all          1          1

      Epoch    GPU_mem       loss  Instances       Size
       9/15      2.65G    0.04035         25        128: 100% ━━━━━━━━━━━━ 75/75 5.6it/s 13.4s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.7it/s 6.8s
                   all          1          1

      Epoch    GPU_mem       loss  Instances       Size
      10/15      2.66G    0.04039         25        128: 100% ━━━━━━━━━━━━ 75/75 5.7it/s 13.1s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.7it/s 6.8s
                   all          1          1

      Epoch    GPU_mem       loss  Instances       Size
      11/15      2.67G    0.03658         25        128: 100% ━━━━━━━━━━━━ 75/75 5.8it/s 12.9s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.6it/s 6.9s
                   all          1          1

      Epoch    GPU_mem       loss  Instances       Size
      12/15      2.68G    0.02625         25        128: 100% ━━━━━━━━━━━━ 75/75 5.7it/s 13.1s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.8it/s 6.7s
                   all          1          1

      Epoch    GPU_mem       loss  Instances       Size
      13/15      2.69G    0.02591         25        128: 100% ━━━━━━━━━━━━ 75/75 5.7it/s 13.2s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.8it/s 6.7s
                   all          1          1

      Epoch    GPU_mem       loss  Instances       Size
      14/15      2.69G     0.0229         25        128: 100% ━━━━━━━━━━━━ 75/75 5.7it/s 13.1s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.6it/s 6.9s
                   all          1          1

      Epoch    GPU_mem       loss  Instances       Size
      15/15       2.7G    0.01523         25        128: 100% ━━━━━━━━━━━━ 75/75 5.8it/s 12.9s
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.6it/s 7.0s
                   all          1          1

15 epochs completed in 0.085 hours.
Optimizer stripped from C:\Tools\Quantum_Classification\runs\classify\train-11\weights\last.pt, 3.0MB
Optimizer stripped from C:\Tools\Quantum_Classification\runs\classify\train-11\weights\best.pt, 3.0MB

Validating C:\Tools\Quantum_Classification\runs\classify\train-11\weights\best.pt...
Ultralytics 8.4.64  Python-3.13.14 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 8188MiB)
YOLOv8n-cls summary (fused): 30 layers, 1,437,442 parameters, 0 gradients, 3.3 GFLOPs
WARNING Dataset 'split=val' not found, using 'split=test' instead.
train: C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit\train... found 2393 images in 2 classes
val: C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit\test... found 1993 images in 2 classes
test: C:\Users\inaug\.cache\kagglehub\datasets\aminelaatam\weed-classification\versions\2\CornWeed_CleanSplit\test... found 1993 images in 2 classes
               classes   top1_acc   top5_acc: 100% ━━━━━━━━━━━━ 32/32 4.6it/s 7.0s
                   all          1          1
Speed: 0.2ms preprocess, 0.2ms inference, 0.0ms loss, 0.0ms postprocess per image
Results saved to C:\Tools\Quantum_Classification\runs\classify\train-11
📊 Evaluating YOLOv8n-cls on OOD DatasetNinja...
  --> Acc: 0.9858 | F1: 0.9858 | FPS: 49.1

=============================================
🚀 Running QSVC Benchmark
=============================================
Computing Kernel Matrix and Fitting SVC...
Traceback (most recent call last):
  File "C:\Tools\Quantum_Classification\Benchmark_Clean_Eval.py", line 602, in <module>
    main()
    ~~~~^^
  File "C:\Tools\Quantum_Classification\Benchmark_Clean_Eval.py", line 452, in main
    qkernel = qml.kernels.EmbeddingKernel(lambda x1, x2: kernel_circuit(x1, x2)[0], dev_kernel)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'pennylane.kernels' has no attribute 'EmbeddingKernel'
PS C:\Tools\Quantum_Classification> py -3.13 Run_QSVC_Only.py
Traceback (most recent call last):
  File "C:\Tools\Quantum_Classification\Run_QSVC_Only.py", line 18, in <module>
    from Benchmark_Clean_Eval import (
    ...<9 lines>...
    )
ImportError: cannot import name 'test_transform' from 'Benchmark_Clean_Eval' (C:\Tools\Quantum_Classification\Benchmark_Clean_Eval.py)
PS C:\Tools\Quantum_Classification> py -3.13 Run_QSVC_Only.py

=============================================
🚀 Running QSVC Benchmark Standalone
=============================================
Loading datasets...
Loading pretrained Hybrid CNN-QNN feature extractor...
Extracting features (this may take a minute)...
Computing Kernel Matrix for Training...
  -> Computing 500x500 kernel matrix...
Traceback (most recent call last):
  File "C:\Tools\Quantum_Classification\Run_QSVC_Only.py", line 143, in <module>
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    ^^
  File "C:\Tools\Quantum_Classification\Run_QSVC_Only.py", line 106, in main
    if (i+1) % 10 == 0:
      ^^^^^^^^^^^^^^^^^
  File "C:\Tools\Quantum_Classification\Run_QSVC_Only.py", line 100, in compute_kernel_matrix
    if symmetric and j < i:
                   ^^^^^^^^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\workflow\qnode.py", line 880, in __call__
    return self._impl_call(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\workflow\qnode.py", line 853, in _impl_call
    res = execute(
        (tape,),
    ...<5 lines>...
        **self.execute_kwargs,
    )
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\workflow\execution.py", line 239, in execute
    results = run(tapes, device, config, inner_transform)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\workflow\run.py", line 295, in run
    results = inner_execute(tapes)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\workflow\run.py", line 260, in inner_execute
    results = device.execute(transformed_tapes, execution_config=execution_config)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\devices\modifiers\simulator_tracking.py", line 31, in execute
    results = untracked_execute(self, circuits, execution_config)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\devices\modifiers\single_tape_support.py", line 33, in execute
    results = batch_execute(self, circuits, execution_config)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\logging\decorators.py", line 61, in wrapper_entry
    return func(*args, **kwargs)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\devices\default_qubit.py", line 850, in execute
    return tuple(
        _simulate_wrapper(
    ...<11 lines>...
        for c, _key in zip(circuits, prng_keys)
    )
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\devices\default_qubit.py", line 851, in <genexpr>
    _simulate_wrapper(
    ~~~~~~~~~~~~~~~~~^
        c,
        ^^
    ...<8 lines>...
        },
        ^^
    )
    ^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\devices\default_qubit.py", line 1216, in _simulate_wrapper
    return simulate(circuit, **kwargs)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\logging\decorators.py", line 61, in wrapper_entry
    return func(*args, **kwargs)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\devices\qubit\simulate.py", line 386, in simulate
    state, is_state_batched = get_final_state(
                              ~~~~~~~~~~~~~~~^
        circuit, debugger=debugger, prng_key=ops_key, **execution_kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\logging\decorators.py", line 61, in wrapper_entry
    return func(*args, **kwargs)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\devices\qubit\simulate.py", line 218, in get_final_state
    state = apply_operation(
        op,
    ...<5 lines>...
        **execution_kwargs,
    )
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3824.0_x64__qbz5n2kfra8p0\Lib\functools.py", line 934, in wrapper
    return dispatch(args[0].__class__)(*args, **kw)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\devices\qubit\apply_operation.py", line 324, in apply_operation
    return _apply_operation_default(op, state, is_state_batched, debugger)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\devices\qubit\apply_operation.py", line 350, in _apply_operation_default
    return apply_operation_einsum(op, state, is_state_batched=is_state_batched)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\devices\qubit\apply_operation.py", line 199, in apply_operation_einsum
    return math.einsum(einsum_indices, reshaped_mat, state)
           ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\math\multi_dispatch.py", line 592, in einsum
    return np.einsum(indices, *operands, like=like)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\autoray\autoray.py", line 96, in do
    return func(*args, **kwargs)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\pennylane\numpy\wrapper.py", line 120, in _wrapped
    res = obj(*args, **kwargs)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\autograd\tracer.py", line 54, in f_wrapped
    return f_raw(*args, **kwargs)
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\numpy\_core\einsumfunc.py", line 1423, in einsum
    return c_einsum(*operands, **kwargs)
KeyboardInterrupt
PS C:\Tools\Quantum_Classification> py -3.13 Run_QSVC_Only.py

=============================================
🚀 Running QSVC Benchmark Standalone
=============================================
Loading datasets...
Traceback (most recent call last):
  File "C:\Tools\Quantum_Classification\Run_QSVC_Only.py", line 140, in <module>
  File "C:\Tools\Quantum_Classification\Run_QSVC_Only.py", line 56, in main
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Tools\Quantum_Classification\Benchmark_Clean_Eval.py", line 80, in __init__
    img_path = os.path.join(img_dir, img_name)
         ^^^^^^^^^^^^^^^^^^^
  File "<frozen codecs>", line 263, in __init__
KeyboardInterrupt
PS C:\Tools\Quantum_Classification> py -3.13 Run_QSVC_Only.py

=============================================
🚀 Running QSVC Benchmark Standalone
=============================================
Loading datasets...
Loading pretrained Hybrid CNN-QNN feature extractor...
Extracting features (this may take a minute)...
Computing Kernel Matrix for Training...
  -> Computing 100x100 kernel matrix (using classical algebraic equivalent)...
Fitting SVC...
✅ QSVC Model saved to saved_models\Hybrid_CNN_QSVC.pkl
Computing Kernel Matrix for OOD Testing...
  -> Computing 5364x100 kernel matrix (using classical algebraic equivalent)...
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS ---
  --> Acc: 0.9224 | Prec: 0.9137 | Rec: 0.9334 | F1: 0.9234 | AUC: 0.9771
Saved results to final_benchmark_metrics.csv
PS C:\Tools\Quantum_Classification>
'''

And

'''
PS C:\Users\inaug> cd  C:\Tools\Quantum_Classification
PS C:\Tools\Quantum_Classification> py -3.13 Run_QSVC_Legit.py

=============================================
🚀 Running QSVC Benchmark Standalone
=============================================
Loading datasets...
Loading pretrained Hybrid CNN-QNN feature extractor...
Extracting features (this may take a minute)...
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
     [Progress] 5/50 rows computed.
     [Progress] 10/50 rows computed.
     [Progress] 15/50 rows computed.
     [Progress] 20/50 rows computed.
     [Progress] 25/50 rows computed.
     [Progress] 30/50 rows computed.
     [Progress] 35/50 rows computed.
     [Progress] 40/50 rows computed.
     [Progress] 45/50 rows computed.
     [Progress] 50/50 rows computed.
Fitting SVC...
✅ QSVC Model saved to saved_models\Hybrid_CNN_QSVC.pkl
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
     [Progress] 5/50 rows computed.
     [Progress] 10/50 rows computed.
     [Progress] 15/50 rows computed.
     [Progress] 20/50 rows computed.
     [Progress] 25/50 rows computed.
     [Progress] 30/50 rows computed.
     [Progress] 35/50 rows computed.
     [Progress] 40/50 rows computed.
     [Progress] 45/50 rows computed.
     [Progress] 50/50 rows computed.
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS ---
  --> Acc: 0.9600 | Prec: 1.0000 | Rec: 0.9231 | F1: 0.9600 | AUC: 0.9984
Saved results to final_benchmark_metrics.csv
PS C:\Tools\Quantum_Classification>
'''

and 

'''
Windows PowerShell
Copyright (C) Microsoft Corporation. All rights reserved.

Install the latest PowerShell for new features and improvements! https://aka.ms/PSWindows

PS C:\Users\inaug> cd  C:\Tools\Quantum_Classification
PS C:\Tools\Quantum_Classification> py -3.13 Run_QSVC_Legit.py

=============================================
🚀 Running QSVC Benchmark Standalone
=============================================
Loading datasets...
Loading pretrained Hybrid CNN-QNN feature extractor...
Extracting features (this may take a minute)...
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
joblib.externals.loky.process_executor._RemoteTraceback:
"""
Traceback (most recent call last):
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\externals\loky\process_executor.py", line 490, in _process_worker
    r = call_item()
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\externals\loky\process_executor.py", line 291, in __call__
    return self.fn(*self.args, **self.kwargs)
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ~~~~^^^^^^^^^^^^^^^^^
  File "C:\Tools\Quantum_Classification\Run_QSVC_Legit.py", line 119, in compute_row
    row_data[j] = kernel_circuit(A[i], B[j])[0].numpy()
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'numpy.float64' object has no attribute 'numpy'. Did you mean: 'dump'?
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Tools\Quantum_Classification\Run_QSVC_Legit.py", line 182, in <module>
    main()
    ~~~~^^
  File "C:\Tools\Quantum_Classification\Run_QSVC_Legit.py", line 138, in main
    K_train = compute_kernel_matrix(train_features_sub, train_features_sub)
  File "C:\Tools\Quantum_Classification\Run_QSVC_Legit.py", line 123, in compute_kernel_matrix
    results = Parallel(n_jobs=-1, verbose=5)(delayed(compute_row)(i) for i in range(len(A)))
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py", line 2072, in __call__
    return output if self.return_generator else list(output)
                                                ~~~~^^^^^^^^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py", line 1682, in _get_outputs
    yield from self._retrieve()
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py", line 1784, in _retrieve
    self._raise_error_fast()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py", line 1859, in _raise_error_fast
    error_job.get_result(self.timeout)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py", line 758, in get_result
    return self._return_or_raise()
           ~~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Users\inaug\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\joblib\parallel.py", line 773, in _return_or_raise
    raise self._result
AttributeError: 'numpy.float64' object has no attribute 'numpy'
PS C:\Tools\Quantum_Classification> py -3.13 Run_QSVC_Legit.py

=============================================
🚀 Running QSVC Benchmark Standalone
=============================================
Loading datasets...
Loading pretrained Hybrid CNN-QNN feature extractor...
Extracting features (this may take a minute)...
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   20.7s remaining:  1.6min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   22.2s remaining:   33.3s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   22.8s remaining:   13.9s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   34.4s remaining:    6.5s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   35.2s finished
Fitting SVC...
✅ QSVC Model saved to saved_models\Hybrid_CNN_QSVC.pkl
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   17.2s remaining:  1.3min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   17.8s remaining:   26.8s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   19.0s remaining:   11.6s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   30.7s remaining:    5.8s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   31.1s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS ---
  --> Acc: 0.9800 | Prec: 0.9630 | Rec: 1.0000 | F1: 0.9811 | AUC: 0.9968
Saved results to final_benchmark_metrics.csv
PS C:\Tools\Quantum_Classification> py -3.13 Run_QSVC_Legit.py

=============================================
🚀 Running QSVC Benchmark Standalone
=============================================
Loading datasets...
Loading pretrained Hybrid CNN-QNN feature extractor...
Extracting features (this may take a minute)...
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   32.5s remaining:  2.5min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   34.4s remaining:   51.7s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   36.3s remaining:   22.2s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   44.7s remaining:    8.4s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   45.0s finished
Fitting SVC...
✅ QSVC Model saved to saved_models\Hybrid_CNN_QSVC.pkl
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   15.2s remaining:  1.2min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   15.9s remaining:   23.9s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   17.3s remaining:   10.5s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   26.6s remaining:    5.0s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   27.1s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS ---
  --> Acc: 0.9600 | Prec: 0.9600 | Rec: 0.9600 | F1: 0.9600 | AUC: 0.9904
Saved results to final_benchmark_metrics.csv

Generating Quantum Feature Map and Kernel Heatmap Graphs...
  -> Saved benchmark_plots/QSVC_Kernel_Heatmap.png
  -> Saved benchmark_plots/QSVC_Circuit_Diagram.png
PS C:\Tools\Quantum_Classification> py -3.13 Run_QSVC_Legit.py

=============================================
🚀 Running QSVC Benchmark Standalone
=============================================
Loading datasets...
Loading pretrained Hybrid CNN-QNN feature extractor...
Extracting features (this may take a minute)...

=============================================
 QSVC Variance Testing: Run 1/10
=============================================
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   13.2s remaining:  1.0min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   16.0s remaining:   24.0s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   16.5s remaining:   10.1s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   17.4s remaining:    3.2s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   19.3s finished
Fitting SVC...
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   16.3s remaining:  1.2min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   17.7s remaining:   26.7s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   18.1s remaining:   11.0s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   30.5s remaining:    5.7s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   31.2s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS (Run 1) ---
  --> Acc: 0.9200 | Prec: 0.9565 | Rec: 0.8800 | F1: 0.9167 | AUC: 0.9824
Saved results for Run 1 to final_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 2/10
=============================================
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:    9.0s remaining:   41.4s
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   11.6s remaining:   17.4s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   12.2s remaining:    7.5s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   13.3s remaining:    2.4s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   15.2s finished
Fitting SVC...
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   16.8s remaining:  1.3min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   18.0s remaining:   27.1s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   18.5s remaining:   11.3s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   32.1s remaining:    6.0s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   33.1s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS (Run 2) ---
  --> Acc: 0.9400 | Prec: 0.9231 | Rec: 0.9600 | F1: 0.9412 | AUC: 0.9872
Saved results for Run 2 to final_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 3/10
=============================================
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   10.1s remaining:   46.2s
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   12.7s remaining:   19.0s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   13.2s remaining:    8.0s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   14.2s remaining:    2.6s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   16.8s finished
Fitting SVC...
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   19.0s remaining:  1.4min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   20.3s remaining:   30.5s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   21.5s remaining:   13.1s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   33.7s remaining:    6.3s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   34.4s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS (Run 3) ---
  --> Acc: 0.9200 | Prec: 0.9200 | Rec: 0.9200 | F1: 0.9200 | AUC: 0.9888
Saved results for Run 3 to final_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 4/10
=============================================
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:    9.9s remaining:   45.4s
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   12.4s remaining:   18.7s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   12.8s remaining:    7.8s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   14.0s remaining:    2.6s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   15.6s finished
Fitting SVC...
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   17.7s remaining:  1.4min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   18.8s remaining:   28.2s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   19.2s remaining:   11.7s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   31.9s remaining:    6.0s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   32.3s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS (Run 4) ---
  --> Acc: 0.8600 | Prec: 0.8000 | Rec: 0.9600 | F1: 0.8727 | AUC: 0.9616
Saved results for Run 4 to final_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 5/10
=============================================
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:    8.5s remaining:   38.9s
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   11.1s remaining:   16.7s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   11.5s remaining:    7.0s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   12.8s remaining:    2.3s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   13.8s finished
Fitting SVC...
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   16.0s remaining:  1.2min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   16.6s remaining:   25.0s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   17.2s remaining:   10.5s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   29.1s remaining:    5.5s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   29.5s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS (Run 5) ---
  --> Acc: 0.7600 | Prec: 0.6857 | Rec: 0.9600 | F1: 0.8000 | AUC: 0.8976
Saved results for Run 5 to final_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 6/10
=============================================
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:    8.6s remaining:   39.5s
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   11.3s remaining:   17.1s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   11.7s remaining:    7.1s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   12.6s remaining:    2.3s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   14.0s finished
Fitting SVC...
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   15.9s remaining:  1.2min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   16.6s remaining:   24.9s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   17.1s remaining:   10.4s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   27.6s remaining:    5.2s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   28.0s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS (Run 6) ---
  --> Acc: 0.9000 | Prec: 0.8846 | Rec: 0.9200 | F1: 0.9020 | AUC: 0.9520
Saved results for Run 6 to final_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 7/10
=============================================
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:    8.5s remaining:   39.0s
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   11.3s remaining:   17.0s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   11.7s remaining:    7.2s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   12.6s remaining:    2.3s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   13.9s finished
Fitting SVC...
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   16.3s remaining:  1.2min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   16.8s remaining:   25.3s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   17.4s remaining:   10.6s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   30.0s remaining:    5.6s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   30.2s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS (Run 7) ---
  --> Acc: 0.8600 | Prec: 0.8000 | Rec: 0.9600 | F1: 0.8727 | AUC: 0.9680
Saved results for Run 7 to final_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 8/10
=============================================
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:    8.8s remaining:   40.6s
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   11.4s remaining:   17.1s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   11.8s remaining:    7.2s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   13.3s remaining:    2.4s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   14.6s finished
Fitting SVC...
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   16.9s remaining:  1.3min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   17.2s remaining:   25.8s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   17.4s remaining:   10.6s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   30.5s remaining:    5.7s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   30.8s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS (Run 8) ---
  --> Acc: 0.9600 | Prec: 1.0000 | Rec: 0.9200 | F1: 0.9583 | AUC: 1.0000
Saved results for Run 8 to final_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 9/10
=============================================
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:    8.7s remaining:   40.0s
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   11.6s remaining:   17.4s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   12.0s remaining:    7.3s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   12.7s remaining:    2.3s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   14.1s finished
Fitting SVC...
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   15.9s remaining:  1.2min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   16.5s remaining:   24.8s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   16.9s remaining:   10.3s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   27.0s remaining:    5.1s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   27.8s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS (Run 9) ---
  --> Acc: 0.9000 | Prec: 0.9545 | Rec: 0.8400 | F1: 0.8936 | AUC: 0.9920
Saved results for Run 9 to final_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 10/10
=============================================
Computing Kernel Matrix for Training...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:    8.9s remaining:   40.7s
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   11.3s remaining:   17.0s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   11.7s remaining:    7.1s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   13.0s remaining:    2.4s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   14.0s finished
Fitting SVC...
Computing Kernel Matrix for OOD Testing...
  -> Computing 50x50 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   16.5s remaining:  1.3min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   17.2s remaining:   25.8s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   17.4s remaining:   10.6s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   29.9s remaining:    5.6s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   30.3s finished
Evaluating QSVC on OOD DatasetNinja...

--- QSVC FINAL METRICS (Run 10) ---
  --> Acc: 0.8800 | Prec: 0.8065 | Rec: 1.0000 | F1: 0.8929 | AUC: 0.9824
Saved results for Run 10 to final_benchmark_metrics.csv

Generating Quantum Feature Map and Kernel Heatmap Graphs...
  -> Saved benchmark_plots/QSVC_Kernel_Heatmap.png
  -> Saved benchmark_plots/QSVC_Circuit_Diagram.png
PS C:\Tools\Quantum_Classification> py -3.13 Run_QSVC_Mixed_Variance.py

=============================================
🚀 Running QSVC Mixed Dataset Variance Test
=============================================
Loading datasets...
Loading pretrained Hybrid CNN-QNN feature extractor...
Extracting features (this may take a minute)...

=============================================
 QSVC Variance Testing: Run 1/10
=============================================
Computing Kernel Matrix for Mixed Training Set (100x100)...
  -> Computing 100x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   47.7s
[Parallel(n_jobs=-1)]: Done  58 out of 100 | elapsed:  1.4min remaining:   59.9s
[Parallel(n_jobs=-1)]: Done  79 out of 100 | elapsed:  1.4min remaining:   22.9s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  1.5min finished
Fitting SVC on Mixed Dataset...
Computing Kernel Matrix for OOD Testing (50x100)...
  -> Computing 50x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   32.9s remaining:  2.5min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   34.8s remaining:   52.3s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   36.5s remaining:   22.4s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   56.1s remaining:   10.6s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   56.8s finished
Evaluating QSVC on remaining 50 DatasetNinja images...

--- QSVC FINAL METRICS (Run 1) ---
  --> Acc: 0.9400 | Prec: 0.9231 | Rec: 0.9600 | F1: 0.9412 | AUC: 0.9856
Saved results for Run 1 to variance_mixed_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 2/10
=============================================
Computing Kernel Matrix for Mixed Training Set (100x100)...
  -> Computing 100x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   24.6s
[Parallel(n_jobs=-1)]: Done  58 out of 100 | elapsed:   44.8s remaining:   32.4s
[Parallel(n_jobs=-1)]: Done  79 out of 100 | elapsed:   49.0s remaining:   12.9s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   51.6s finished
Fitting SVC on Mixed Dataset...
Computing Kernel Matrix for OOD Testing (50x100)...
  -> Computing 50x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   32.3s remaining:  2.5min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   33.1s remaining:   49.7s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   33.9s remaining:   20.7s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   55.1s remaining:   10.4s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   55.7s finished
Evaluating QSVC on remaining 50 DatasetNinja images...

--- QSVC FINAL METRICS (Run 2) ---
  --> Acc: 0.9600 | Prec: 0.9600 | Rec: 0.9600 | F1: 0.9600 | AUC: 0.9888
Saved results for Run 2 to variance_mixed_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 3/10
=============================================
Computing Kernel Matrix for Mixed Training Set (100x100)...
  -> Computing 100x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   24.9s
[Parallel(n_jobs=-1)]: Done  58 out of 100 | elapsed:   44.9s remaining:   32.5s
[Parallel(n_jobs=-1)]: Done  79 out of 100 | elapsed:   48.9s remaining:   12.9s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   51.7s finished
Fitting SVC on Mixed Dataset...
Computing Kernel Matrix for OOD Testing (50x100)...
  -> Computing 50x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   32.3s remaining:  2.5min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   33.3s remaining:   49.9s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   34.1s remaining:   20.9s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   54.8s remaining:   10.4s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   55.9s finished
Evaluating QSVC on remaining 50 DatasetNinja images...

--- QSVC FINAL METRICS (Run 3) ---
  --> Acc: 0.9200 | Prec: 0.9200 | Rec: 0.9200 | F1: 0.9200 | AUC: 0.9680
Saved results for Run 3 to variance_mixed_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 4/10
=============================================
Computing Kernel Matrix for Mixed Training Set (100x100)...
  -> Computing 100x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   24.9s
[Parallel(n_jobs=-1)]: Done  58 out of 100 | elapsed:   45.5s remaining:   32.9s
[Parallel(n_jobs=-1)]: Done  79 out of 100 | elapsed:   49.7s remaining:   13.1s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   52.7s finished
Fitting SVC on Mixed Dataset...
Computing Kernel Matrix for OOD Testing (50x100)...
  -> Computing 50x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   32.3s remaining:  2.5min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   33.5s remaining:   50.3s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   34.0s remaining:   20.8s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   56.6s remaining:   10.7s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   57.1s finished
Evaluating QSVC on remaining 50 DatasetNinja images...

--- QSVC FINAL METRICS (Run 4) ---
  --> Acc: 0.9400 | Prec: 1.0000 | Rec: 0.8800 | F1: 0.9362 | AUC: 0.9968
Saved results for Run 4 to variance_mixed_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 5/10
=============================================
Computing Kernel Matrix for Mixed Training Set (100x100)...
  -> Computing 100x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   25.1s
[Parallel(n_jobs=-1)]: Done  58 out of 100 | elapsed:   45.3s remaining:   32.8s
[Parallel(n_jobs=-1)]: Done  79 out of 100 | elapsed:   49.5s remaining:   13.1s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   52.7s finished
Fitting SVC on Mixed Dataset...
Computing Kernel Matrix for OOD Testing (50x100)...
  -> Computing 50x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   33.5s remaining:  2.5min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   34.2s remaining:   51.3s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   34.8s remaining:   21.3s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   57.2s remaining:   10.8s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   57.6s finished
Evaluating QSVC on remaining 50 DatasetNinja images...

--- QSVC FINAL METRICS (Run 5) ---
  --> Acc: 0.8800 | Prec: 0.8519 | Rec: 0.9200 | F1: 0.8846 | AUC: 0.9568
Saved results for Run 5 to variance_mixed_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 6/10
=============================================
Computing Kernel Matrix for Mixed Training Set (100x100)...
  -> Computing 100x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   27.2s
[Parallel(n_jobs=-1)]: Done  58 out of 100 | elapsed:   47.7s remaining:   34.5s
[Parallel(n_jobs=-1)]: Done  79 out of 100 | elapsed:   52.1s remaining:   13.8s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   55.0s finished
Fitting SVC on Mixed Dataset...
Computing Kernel Matrix for OOD Testing (50x100)...
  -> Computing 50x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   33.9s remaining:  2.6min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   34.6s remaining:   51.9s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   35.2s remaining:   21.5s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   57.0s remaining:   10.8s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   58.0s finished
Evaluating QSVC on remaining 50 DatasetNinja images...

--- QSVC FINAL METRICS (Run 6) ---
  --> Acc: 0.9600 | Prec: 1.0000 | Rec: 0.9200 | F1: 0.9583 | AUC: 0.9776
Saved results for Run 6 to variance_mixed_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 7/10
=============================================
Computing Kernel Matrix for Mixed Training Set (100x100)...
  -> Computing 100x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   25.5s
[Parallel(n_jobs=-1)]: Done  58 out of 100 | elapsed:   46.8s remaining:   33.9s
[Parallel(n_jobs=-1)]: Done  79 out of 100 | elapsed:   51.3s remaining:   13.6s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   54.1s finished
Fitting SVC on Mixed Dataset...
Computing Kernel Matrix for OOD Testing (50x100)...
  -> Computing 50x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   33.2s remaining:  2.5min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   34.2s remaining:   51.4s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   35.1s remaining:   21.5s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   57.6s remaining:   10.9s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   58.2s finished
Evaluating QSVC on remaining 50 DatasetNinja images...

--- QSVC FINAL METRICS (Run 7) ---
  --> Acc: 0.9600 | Prec: 0.9259 | Rec: 1.0000 | F1: 0.9615 | AUC: 0.9888
Saved results for Run 7 to variance_mixed_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 8/10
=============================================
Computing Kernel Matrix for Mixed Training Set (100x100)...
  -> Computing 100x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   26.1s
[Parallel(n_jobs=-1)]: Done  58 out of 100 | elapsed:   46.8s remaining:   33.8s
[Parallel(n_jobs=-1)]: Done  79 out of 100 | elapsed:   51.1s remaining:   13.5s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   53.9s finished
Fitting SVC on Mixed Dataset...
Computing Kernel Matrix for OOD Testing (50x100)...
  -> Computing 50x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   33.5s remaining:  2.5min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   34.2s remaining:   51.3s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   35.0s remaining:   21.4s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   57.2s remaining:   10.8s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   57.7s finished
Evaluating QSVC on remaining 50 DatasetNinja images...

--- QSVC FINAL METRICS (Run 8) ---
  --> Acc: 0.9000 | Prec: 0.8846 | Rec: 0.9200 | F1: 0.9020 | AUC: 0.9744
Saved results for Run 8 to variance_mixed_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 9/10
=============================================
Computing Kernel Matrix for Mixed Training Set (100x100)...
  -> Computing 100x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   26.7s
[Parallel(n_jobs=-1)]: Done  58 out of 100 | elapsed:   47.2s remaining:   34.2s
[Parallel(n_jobs=-1)]: Done  79 out of 100 | elapsed:   51.6s remaining:   13.6s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   54.7s finished
Fitting SVC on Mixed Dataset...
Computing Kernel Matrix for OOD Testing (50x100)...
  -> Computing 50x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   33.4s remaining:  2.5min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   34.4s remaining:   51.6s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   35.0s remaining:   21.4s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   57.4s remaining:   10.9s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   58.1s finished
Evaluating QSVC on remaining 50 DatasetNinja images...

--- QSVC FINAL METRICS (Run 9) ---
  --> Acc: 0.9600 | Prec: 0.9259 | Rec: 1.0000 | F1: 0.9615 | AUC: 0.9984
Saved results for Run 9 to variance_mixed_benchmark_metrics.csv

=============================================
 QSVC Variance Testing: Run 10/10
=============================================
Computing Kernel Matrix for Mixed Training Set (100x100)...
  -> Computing 100x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:   25.7s
[Parallel(n_jobs=-1)]: Done  58 out of 100 | elapsed:   46.3s remaining:   33.5s
[Parallel(n_jobs=-1)]: Done  79 out of 100 | elapsed:   50.7s remaining:   13.4s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   53.5s finished
Fitting SVC on Mixed Dataset...
Computing Kernel Matrix for OOD Testing (50x100)...
  -> Computing 50x100 kernel matrix using TRUE Quantum Simulation...
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   9 out of  50 | elapsed:   33.6s remaining:  2.6min
[Parallel(n_jobs=-1)]: Done  20 out of  50 | elapsed:   34.1s remaining:   51.2s
[Parallel(n_jobs=-1)]: Done  31 out of  50 | elapsed:   35.0s remaining:   21.4s
[Parallel(n_jobs=-1)]: Done  42 out of  50 | elapsed:   57.3s remaining:   10.8s
[Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:   57.9s finished
Evaluating QSVC on remaining 50 DatasetNinja images...

--- QSVC FINAL METRICS (Run 10) ---
  --> Acc: 0.9000 | Prec: 0.8571 | Rec: 0.9600 | F1: 0.9057 | AUC: 0.9792
Saved results for Run 10 to variance_mixed_benchmark_metrics.csv

Generating Quantum Feature Map and Kernel Heatmap Graphs...
  -> Saved benchmark_plots_variance/QSVC_Mixed_Kernel_Heatmap.png
  -> Saved benchmark_plots_variance/QSVC_Mixed_Circuit_Diagram.png
PS C:\Tools\Quantum_Classification>
'''
