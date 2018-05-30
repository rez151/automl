import Augmentor

p = Augmentor.Pipeline(source_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/la/500/0_nodr",
                       output_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/equal/train/0_nodr/")
p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
p.zoom(probability=1, min_factor=1.0, max_factor=1.1)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.sample(10000, multi_threaded=True)

p = Augmentor.Pipeline(source_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/la/500/1_mild",
                       output_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/equal/train/1_mild/")
p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
p.zoom(probability=1, min_factor=1.0, max_factor=1.1)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.sample(10000, multi_threaded=True)

p = Augmentor.Pipeline(source_directory="//home/determinants/automl/datasets/diabetic-retinopathy-detection/la/500/2_moderate",
                       output_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/equal/train/2_moderate/")
p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
p.zoom(probability=1, min_factor=1.0, max_factor=1.1)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.sample(10000, multi_threaded=True)

p = Augmentor.Pipeline(source_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/la/500/3_severe",
                       output_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/equal/train/3_severe/")
p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
p.zoom(probability=1, min_factor=1.0, max_factor=1.1)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.sample(10000, multi_threaded=True)

p = Augmentor.Pipeline(source_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/la/500/4_proliferativedr",
                       output_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/equal/train/4_proliferativedr/")
p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
p.zoom(probability=1, min_factor=1.0, max_factor=1.1)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.sample(10000, multi_threaded=True)
