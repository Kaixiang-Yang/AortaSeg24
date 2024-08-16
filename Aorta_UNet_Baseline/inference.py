import torch
from inference_code import predict_from_folder_segrap2023
# from inference_code import convert_one_hot_label_to_multi_lesions
# from evalutils.validators import (
#     UniquePathIndicesValidator,
#     UniqueImagesValidator,
# )
# from evalutils import SegmentationAlgorithm
import os
import sys
o_path = os.getcwd()
print(o_path)
sys.path.append(o_path)

### preprocessed data has to be turned into x, y, z; and meanwhile the save process
class Customalgorithm():  # SegmentationAlgorithm is not inherited in this class anymore
    def __init__(self):
        """
        Do not modify the `self.input_dir` and `self.output_dir`. 
        (Check https://grand-challenge.org/algorithms/interfaces/)
        """
        self.input_dir = "/input/images/ct-angiography/"
        self.output_dir = "/output/images/aortic-branches/"
        # self.input_dir = "/home/yangkaixiang/OrganSeg/nnUNetFrame/AortaSeg24-main/Aorta_UNet_Baseline/input/images/ct-angiography"
        # self.output_dir = "/home/yangkaixiang/OrganSeg/nnUNetFrame/AortaSeg24-main/Aorta_UNet_Baseline/output/images/aortic-branches"

        self.weight = "./resources/"

    def predict(self):
        """
        load the model and checkpoint, and generate the predictions. You can replace this part with your own model.
        """
        predict_from_folder_segrap2023(self.weight, self.input_dir, self.output_dir, 'all', 0, 1)
        print("nnUNet segmentation done!")
        print('Prediction finished !')
    

    def process(self):
        self.predict()

if __name__ == "__main__":
    raise SystemExit(Customalgorithm().process())
