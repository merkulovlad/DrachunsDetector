Create an env, install requirements.

Run demo_video_infer_async.py

python demo_video_infer_async.py \
  --video PATH_TO_VIDEO \
  --clf checkpoints/best_model.pth \
  --weights yolov8n-pose.pt 

you can train your own model in train.ipynb but you need precomputed graphs of your dataset with videos. For this you can use precompute_kaggle.ipynb for doing this on kaggle with free gpu.




Was tested on python 3.13.5, 3.9 doesn't work for sure.