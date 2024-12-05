import cv2
import numpy as np
from config import load_config
import argparse
def enhance_text_clarity(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"图像文件 {image_path} 未找到")

    # 双边滤波
    bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # 自适应直方图均衡化
    lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 应用锐化滤波器
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='conf.yml', help='path to the config.yaml file')
    args = parser.parse_args()
    args.config = './Deblurring/conf.yml'
    config = load_config(args.config)
    print('Config loaded')
    mode = config.MODE
    task = config.TASK
    if task == "Deblurring":
        from Deblurring.src.trainer import Trainer
        from Deblurring.src.tester import Tester
        from Deblurring.src.finetune import Finetune
    else:
        from Binarization.src.trainer import Trainer
        from Binarization.src.tester import Tester

    if mode == 0:
        print("--------------------------")
        print('Start Testing')
        print("--------------------------")

        tester = Tester(config)
        tester.test()

        print("--------------------------")
        print('Testing complete')
        print("--------------------------")

    elif mode == 1:


        print("--------------------------")
        print('Start Training')
        print("--------------------------")

        trainer = Trainer(config)
        trainer.train()

        print("--------------------------")
        print('Training complete')
        print("--------------------------")


    else: 
        print("--------------------------")
        print('Start Finetuning')
        print("--------------------------")

        finetuner = Finetune(config)
        finetuner.finetune()

        print("--------------------------")
        print('Finetuning complete')
        print("--------------------------")


        
if __name__ == "__main__":
    main()