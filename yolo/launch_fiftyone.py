import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("coco-2017", drop_existing_dataset=True)
session = fo.launch_app(dataset, port=5151)
session.wait()
