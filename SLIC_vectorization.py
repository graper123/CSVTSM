# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import cv2
from osgeo import ogr, osr, gdal
import numpy as np
from PIL import Image
from skimage import morphology, color, measure
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.future import graph
from skimage import data, filters
import time
from skimage.morphology import disk




#读取if文件
def read_img(filename):
    dataset = gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    del dataset
    #返回tif为图像的行数、列数、投影信息、仿射矩阵、图像数据
    return im_width, im_height, im_proj, im_geotrans, im_data

#写入tif文件
def write_img(filename, im_proj, im_geotrans, im_data):
    #判断栅格图像格式
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    #判断图像数据的维度
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff") #数据类型
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype) #创建文件

    dataset.SetGeoTransform(im_geotrans) #写入仿射变化参数
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data) #写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


def stretch_n(bands, img_min, img_max, lower_percent=0, higher_percent=100):
    out = np.zeros_like(bands).astype(np.float32)
    a = img_min
    b = img_max
    c = np.percentile(bands[:, :], lower_percent)
    d = np.percentile(bands[:, :], higher_percent)
    t = a + (bands[:, :] - c) * (b - a) / (d - c)
    t[t < a] = a
    t[t > b] = b
    out[:, :] = t
    return out


def DoesDriverHandleExtension(drv, ext):
    exts = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
    return exts is not None and exts.lower().find(ext.lower()) >= 0


def GetExtension(filename):
    ext = os.path.splitext(filename)[1]
    if ext.startswith('.'):
        ext = ext[1:]
    return ext


def GetOutputDriversFor(filename):
    drv_list = []
    ext = GetExtension(filename)
    for i in range(gdal.GetDriverCount()):
        drv = gdal.GetDriver(i)
        if (drv.GetMetadataItem(gdal.DCAP_CREATE) is not None or
            drv.GetMetadataItem(gdal.DCAP_CREATECOPY) is not None) and \
                drv.GetMetadataItem(gdal.DCAP_VECTOR) is not None:
            if ext and DoesDriverHandleExtension(drv, ext):
                drv_list.append(drv.ShortName)
            else:
                prefix = drv.GetMetadataItem(gdal.DMD_CONNECTION_PREFIX)
                if prefix is not None and filename.lower().startswith(prefix.lower()):
                    drv_list.append(drv.ShortName)

    return drv_list


def GetOutputDriverFor(filename):
    drv_list = GetOutputDriversFor(filename)
    ext = GetExtension(filename)
    if not drv_list:
        if not ext:
            return 'ESRI Shapefile'
        else:
            raise Exception("Cannot guess driver for %s" % filename)
    elif len(drv_list) > 1:
        print("Several drivers matching %s extension. Using %s" % (ext if ext else '', drv_list[0]))
    return drv_list[0]


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.
    The method expects that the mean color of `dst` is already computed.
    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.
    This method computes the mean color of `dst`.
    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])


def BetterMedianFilter(src_arr, k=3, padding=None):
    # imarray = np.array(Image.open(src))
    height, width = src_arr.shape

    if not padding:
        edge = int((k - 1) / 2)
        if height - 1 - edge <= edge or width - 1 - edge <= edge:
            print("The parameter k is to large.")
            return None
        new_arr = np.zeros((height, width), dtype="uint16")
        for i in range(height):
            for j in range(width):
                if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= height - edge - 1:
                    new_arr[i, j] = src_arr[i, j]
                else:
                    nm = src_arr[i - edge:i + edge + 1, j - edge:j + edge + 1]
                    max = np.max(nm)
                    min = np.min(nm)
                    if src_arr[i, j] == max or src_arr[i, j] == min:
                        new_arr[i, j] = np.median(nm)
                    else:
                        new_arr[i, j] = src_arr[i, j]

        return new_arr


if __name__ == '__main__':
    img_path = r"D:\meng\test\test.tif"
    temp_path = r"D:\meng\shp\\"
    start = time.time()
    #temp_path = "D:\\swin-transformer\\result\\temp1_quickshift\\"
    # 根据shp文件与相应的tif文件裁剪图像
    im_width, im_height, im_proj, im_geotrans, im_data = read_img(img_path)
    print(im_data.shape)
    im_data = im_data[0:3]
    print(im_data.shape)
    temp = im_data.transpose((2, 1, 0))

    segments_slic = slic(temp, n_segments=1500, compactness=50, max_num_iter=10, sigma=1) # 使用slic对遥感图像进行超像素分割
    mark0 = mark_boundaries(temp,segments_slic) # 返回带有带有边界的标签图像
    save_path = temp_path + "qs_seg0.tif"
    re0 = mark0.transpose((2, 1, 0)) # 将图像的shape改变成能够写入tif文件的格式
    write_img(save_path, im_proj, im_geotrans, re0) # 将带有边界的标签图像写入tif文件

    grid_path = temp_path + "qs_grid0.tif"
    grid0 = np.uint8(re0[0, ...]) #将图像格式转为uint8格式
    write_img(grid_path, im_proj, im_geotrans, grid0) # 将uint8格式的图像写入tif文件
    skeleton = morphology.skeletonize(grid0) # 返回二进制图像的骨架，能将一个连通区域细化成一个像素的宽度，用于特征提取和目标拓扑表示
    border0 = np.multiply(grid0, skeleton) # 将二进制图像的grid0 与二进制图像的骨架的矩阵相乘
    ret, border0 = cv2.threshold(border0, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border_path = temp_path + "qs_border0.tif"
    write_img(border_path, im_proj, im_geotrans, border0)


    g = graph.rag_mean_color(temp,segments_slic)
    labels2 = graph.merge_hierarchical(segments_slic, g, thresh=5,
                                       rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color) # 对邻接图进行分级合并，返回新的标签数组

    label_rgb2 = color.label2rgb(labels2, temp, kind='avg') # 给定标签数组，与原始图像，返回带有标签的RGB图像

    rgb_path = temp_path + "qs_label.tif"
    lb = labels2.transpose((1, 0))
    write_img(rgb_path, im_proj, im_geotrans, lb)

    label_smooth = temp_path + "qs_label_smooth.tif"

    lb = BetterMedianFilter(lb) # 使用中值滤波对线进行平滑
    write_img(label_smooth, im_proj, im_geotrans, lb)

    mark = mark_boundaries(label_rgb2, labels2)
    save_path = temp_path + "qs_seg.tif"
    re = mark.transpose((2, 1, 0))
    write_img(save_path, im_proj, im_geotrans, re)

    grid_path = temp_path + "qs_grid.tif"
    grid = np.uint8(re[0, ...])
    write_img(grid_path, im_proj, im_geotrans, grid)

    skeleton = morphology.skeletonize(grid)
    border = np.multiply(grid, skeleton)
    ret, border = cv2.threshold(border, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border_path = temp_path + "qs_border.tif"
    write_img(border_path, im_proj, im_geotrans, border)



    border_driver = gdal.Open(rgb_path)
    border_band = border_driver.GetRasterBand(1)
    border_mask = border_band.GetMaskBand()

    dst_filename = temp_path + 'temp.shp'
    frmt = GetOutputDriverFor(dst_filename)
    drv = ogr.GetDriverByName(frmt)
    dst_ds = drv.CreateDataSource(dst_filename)

    dst_layername = 'out'
    srs = osr.SpatialReference(wkt=border_driver.GetProjection())
    dst_layer = dst_ds.CreateLayer(dst_layername, geom_type=ogr.wkbPolygon, srs=srs)


    dst_fieldname = 'DN'
    fd = ogr.FieldDefn(dst_fieldname, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 0

    options = [""]
    options.append('DATASET_FOR_GEOREF=' + rgb_path)
    prog_func = gdal.TermProgress_nocb
    gdal.Polygonize(border_band, border_mask, dst_layer, dst_field, options,
                    callback=prog_func)

    srcband = None
    src_ds = None
    dst_ds = None
    mask_ds = None
    end = time.time()
    print('分割所花时间：',(end-start)/60,'min')

