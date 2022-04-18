# dataset path
export VOC_ROOT="./datasets"


# train
python train.py --config-file ./configs/dpssd512_voc0712.yaml


# evaluate
# python test.py --config-file configs/dpssd320_voc0712.yaml


# inference
# python demo.py --config-file configs/dpssd320_voc0712.yaml --images_dir demo --ckpt [ckpt_path]
