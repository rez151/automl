DR_CONFIG = {
    'name': 'drdetection',
    'loaddir': '/home/determinants/automl/datasets/diabetic-retinopathy-detection',
    'labelfile': '/home/determinants/automl/datasets/diabetic-retinopathy-detection/trainLabels.csv',
    'fileextension': '.jpeg',
    'classes': ('0_nodr', '1_mild', '2_moderate', '3_severe', '4_proliferativedr'),
    'classweights': (1, 11, 5, 29, 37),
    'validationsplit': 0.2
}