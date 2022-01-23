cd demo
python demo_model_visual.py 
ffmpeg -r 5 -pattern_type glob -i './demo_infer/*.png' -b 8000k demo_visual.mp4 -pix_fmt yuv420p