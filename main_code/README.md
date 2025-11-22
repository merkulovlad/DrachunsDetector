Create an env, install requirements.

Run demo_video_infer_async.py

python demo_video_infer_async.py \
  --video PATH_TO_VIDEO \
  --clf checkpoints/best_model.pth \
  --weights yolov8n-pose.pt \

you can train your own model in train.ipynb but you need precomputed graphs of your dataset with videos. For this you can use precompute_graphs.ipynb if you have cluster to run it locally or precompute_on_kaggle for doing this on kaggle with free gpu. The difference is that the version of kaggle have the whole code in itself, while local version just import code from the directory.




Was tested on python 3.13.5, 3.9 doesn't work for sure.