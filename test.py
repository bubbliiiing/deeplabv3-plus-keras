#------------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   miou测试请看get_miou_prediction.py、miou.py
#------------------------------------------------#
from nets.deeplab import Deeplabv3

if __name__ == "__main__":
    model = Deeplabv3(21, [512,512,3], backbone='mobilenet')
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
