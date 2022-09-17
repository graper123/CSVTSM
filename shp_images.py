import os
from osgeo import ogr, osr, gdal

# img_path = "D:\\swin-transformer\\result\\area.tif"
# temp_path = "D:\\swin-transformer\\result\\temp"
#根据shp文件与相应的tif文件裁剪图像
def read_img(filename):
    dataset=gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)

    del dataset
    return im_proj,im_geotrans,im_width, im_height,im_data


def write_img(filename, im_proj, im_geotrans, im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset


#存结果版（不带坐标系）
# converts coordinates to index
def bbox2ix(bbox,gt):
    xo = int(round((bbox[0] - gt[0])/gt[1]))
    yo = int(round((gt[3] - bbox[3])/gt[1]))
    xd = int(round((bbox[1] - bbox[0])/gt[1]))
    yd = int(round((bbox[3] - bbox[2])/gt[1]))
    return(xo,yo,xd,yd)

def rasclip(ras,shp,save):
    ds = gdal.Open(ras)
    gt = ds.GetGeoTransform()
    pj = ds.GetProjection()

    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shp, 0)
    layer = dataSource.GetLayer()


    count = 0
    for feature in layer:
        xo,yo,xd,yd = bbox2ix(feature.GetGeometryRef().GetEnvelope(),gt)
        arr = ds.ReadAsArray(xo,yo,xd,yd)
        save_path =os.path.join(save, str(count)+'.tif')
        write_img(save_path, pj, gt, arr)
        count += 1

    layer.ResetReading()
    ds = None
    dataSource = None

if __name__ == "__main__":
    ras = r"H:\CL\1\2.tif"
    shp = r"H:\CL\shp\2\\temp.shp"
    save = r"H:\CL\area"
    rasclip(ras, shp, save)
