from ultralytics import YOLOv10
import warnings
warnings.filterwarnings('ignore')
# 模型配置文件
#model_yaml_path = ""
#数据集配置文件
data_yaml_path = ''
#预训练模型
pre_model_name = ''  #Pruned weight file as pre-trained model
if __name__ == '__main__':
    # #加载预训练模型
    model = YOLOv10(pre_model_name)
    # 不加载预训练模型
    #model = YOLOv10(model_yaml_path)
    #训练模型
    results = model.train(data=data_yaml_path,
                          imgsz=640,
                          epochs=150,
                          batch=1,
                          workers=1,
                          optimizer='SGD',  # using SGD
                          amp=False,  # 如果出现训练损失为Nan可以关闭amp
                          project='runs/V10train',
                          name='exp',
                          )