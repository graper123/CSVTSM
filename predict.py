import os
import json
import time
import torch
from PIL import Image
from torchvision import transforms
from osgeo import gdal,ogr
from model import convnext_tiny as create_model
import xlsxwriter as xw
import pandas as pd



# 计算图层中所有要素的面积并写入shapefile，并返回总面积
def area(shp_path):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    driver = ogr.Open(shp_path,1)
    layer = driver.GetLayer()
    newfield = ogr.FieldDefn('Area1',ogr.OFTReal)
    newfield.SetWidth(32)
    newfield.SetPrecision(16)
    layer.CreateField(newfield)
    area_num = 0
    for feature in layer:
        geom = feature.GetGeometryRef()
        geom2 = geom.Clone()
        # geom2.Transform(geom)
        area = geom2.GetArea()
        feature.SetField('Area1',area)
        area_num += area
        layer.SetFeature(feature)
    print('各个地物面积计算成功！','地物总面积为：',area_num)
    driver.Destroy()
    return area_num

# 按照要求计算某一类别的总面积，并返回该类别的总面积num
def total_area(shp_path):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    driver = ogr.Open(shp_path, 0)
    layer = driver.GetLayer()
    layer.SetAttributeFilter("tiny2 = 6")
    num = 0
    for feature in layer:
        num += feature.GetField('Area1')
    print('total area of class_5 is :',num)
    driver.Destroy()
    return num

# 将数据写入excel
def xw_toExcel(data, fileName):  # xlsxwriter库储存数据到excel
    workbook = xw.Workbook(fileName)  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    title = ['FID', 'type']  # 设置表头
    worksheet1.write_row('A1', title)  # 从A1单元格开始写入表头
    i = 2  # 从第二行开始写入数据
    for j in range(len(data)):
        insertData = [j, data[j]]
        row = 'A' + str(i)
        worksheet1.write_row(row, insertData)
        i += 1
    workbook.close()  # 关闭表
    print('文件写入成功！')


def get_img_path(img_path):
    imgList = os.listdir(img_path)
    imgList.sort(key=lambda x: int(x.split('.')[0]))
    #print(imgList)
    imgpath_list = []
    for count in range(0, len(imgList)):
        img_name = imgList[count]
        img_path1 = os.path.join( img_path,img_name)
        imgpath_list.append(img_path1)
    #print(imgpath_list)
    return imgpath_list
def data_prosessing(img):
    data_transform = transforms.Compose([transforms.Resize(int(224 * 1.143)),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return data_transform(img)

def load_image(train_path):
    img_path_list = get_img_path(train_path)
    #path_list = path(train_path)
    img_list = []
    for img_path in img_path_list:
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = data_prosessing(img)
        img_list.append(img)
    return img_list
def predict(img_list,model_path):
    i = 0
    result = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for image1 in img_list:
        # print(image1)
        image1 = torch.unsqueeze(image1, dim=0)
        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = create_model(num_classes=9).to(device)
        # load model weights
        model_weight_path = model_path
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(image1.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        result.append(predict_cla)
        print(i, ':', predict_cla)
        i = i + 1
    return result
def write_shapefile(shapefile_path,str,excel_path,):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data = driver.Open(shapefile_path,1)
    layer = data.GetLayer(0)
    result = readexcel(excel_path)
    n = 0
    # for i in range(len(result)):
    #     print(result[i])
    for feat in layer:
        feat.SetField(str,result[n])
        n += 1
        layer.SetFeature(feat)
    layer.ResetReading()
    print('shapefile属性写入成功')
# 创建新的字段
def create_shapefilename(str,shapefile_path):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data = driver.Open(shapefile_path,1)
    layer = data.GetLayer(0)
    layer.CreateField(ogr.FieldDefn(str, ogr.OFTInteger))
    print('创建字段成功')
def readexcel(path):
    # filename = xlrd.open_workbook(path)
    # table = filename.sheets()[0]
    # nrows = table.nrows()
    # ncols = table.ncols()
    # result = table.col_values(2)
    # return result
    df = pd.read_excel(path,usecols=[1],names=None)
    df_li = df.values.tolist()
    result = []
    for i in df_li:
        result.append(i[0])
    return result

def main():
    train_path = r'D:\树种制图\or_image of data\area'
    model_path = r'D:\语义分割\deep-learning-for-image-processing-master\pytorch_classification\ConvNeXt\weights\best_model_NEWDATA.pth'
    img_list = load_image(train_path)
    return predict(img_list,model_path)

if __name__ == '__main__':
    timestart = time.time()
    xx = main()
    shapefilepath = r'D:\树种制图\shp_area\SLIC\area\temp.shp'
    filepath =r"D:\树种制图\result\arcgis_Caijian\area.xlsx"
    str1 = 'Conv'
    xw_toExcel(xx,filepath)
    create_shapefilename(str1,shapefilepath)
    write_shapefile(shapefilepath,str1,excel_path=filepath)
    # # area_num = area(shapefilepath)
    # area1 = total_area(shapefilepath)
    endtime = time.time()
    time_sum = endtime-timestart
    print('预测总共花费',time_sum / 60,'min')
