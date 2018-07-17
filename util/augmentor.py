import Augmentor

print("1")
p = Augmentor.Pipeline(source_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/preprocess/512/train/0_nodr",
                       output_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/equal/train/0_nodr/")
p.zoom(probability=1, min_factor=1, max_factor=1.1)
p.flip_top_bottom(probability=0.5)
p.sample(10000, multi_threaded=True)

print("2")
p = Augmentor.Pipeline(source_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/preprocess/512/train/1_mild",
                       output_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/equal/train/1_mild/")
p.zoom(probability=1, min_factor=1, max_factor=1.1)
p.flip_top_bottom(probability=0.5)
p.sample(10000, multi_threaded=True)

print("3")
p = Augmentor.Pipeline(source_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/preprocess/512/train/2_moderate",
                       output_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/equal/train/2_moderate/")
p.zoom(probability=1, min_factor=1, max_factor=1.1)
p.flip_top_bottom(probability=0.5)
p.sample(10000, multi_threaded=True)

print("4")
p = Augmentor.Pipeline(source_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/preprocess/512/train/3_severe",
                       output_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/equal/train/3_severe/")
p.zoom(probability=1, min_factor=1, max_factor=1.1)
p.flip_top_bottom(probability=0.5)
p.sample(10000, multi_threaded=True)

print("5")
p = Augmentor.Pipeline(source_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/preprocess/512/train/4_proliferativedr",
                       output_directory="/home/determinants/automl/datasets/diabetic-retinopathy-detection/equal/train/4_proliferativedr/")
p.zoom(probability=1, min_factor=1, max_factor=1.1)
p.flip_top_bottom(probability=0.5)
p.sample(10000, multi_threaded=True)

