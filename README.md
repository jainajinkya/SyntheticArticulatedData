# SyntheticArticulatedData
Procedurally generated articulated objects specified in Universal Robot Description Format (URDF), and rendered using Mujoco.

Setup:
```pip install -r requirements.txt```

Example generation:
```python generate_data.py --n 10 --dir ./test --obj microwave --masked --debug```

### Sapien Dataset Generation
Check if the dataset contains the correct mujoco-compatible urdf file or not. If yes, skip to next step, else run:
```
bash ./sapien_dataset/obj_to_mujoco_xml_converter.sh ~/datasets/partnet-mobility-dataset/7119-test/ microwave
```

Generate dataset
```
python generate_data_sapien.py --n 10 --dir ../data/debug/mw_sapien --obj microwave --obj-xml-file ~/datasets/partnet-mobility-dataset/7119-test/mobility_mujoco.xml --masked --debug
```