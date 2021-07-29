
###读取envi遥感数据
import gdal
from skimage import io
import scipy.io as sio

class ENVI_read:
    def __init__(self, in_file):
        self.in_file = in_file  # Tiff或者ENVI文件
        dataset = gdal.Open(self.in_file)
        self.XSize = dataset.RasterXSize  # 网格的X轴像素数量
        self.YSize = dataset.RasterYSize  # 网格的Y轴像素数量
        self.GeoTransform = dataset.GetGeoTransform()  # 投影转换信息
        self.ProjectionInfo = dataset.GetProjection()  # 投影信息
        self.im_bands = dataset.RasterCount
    #band: 读取第几个通道的数据
    def get_data(self, band):
            
        dataset = gdal.Open(self.in_file)
        band = dataset.GetRasterBand(band)
        data = band.ReadAsArray()
        return data
    #获取经纬度信息
    def get_lon_lat(self):
        gtf = self.GeoTransform
        x_range = range(0, self.XSize)
        y_range = range(0, self.YSize)
        x, y = np.meshgrid(x_range, y_range)
        lon = gtf[0] + x * gtf[1] + y * gtf[2]
        lat = gtf[3] + x * gtf[4] + y * gtf[5]
        return lon, lat
class standard_read:
    #IMREAD_UNCHANGED = -1#不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
    # IMREAD_GRAYSCALE = 0#进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
    # IMREAD_COLOR = 1#进行转化为RGB三通道图像，图像深度转为8位
    # IMREAD_ANYDEPTH = 2#保持图像深度不变，进行转化为灰度图。
    # IMREAD_ANYCOLOR = 4#若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位
    def __init__(self,in_file):
        self.in_file = in_file  # Tiff或者ENVI文件
    def mat_read(self):
        matfile = sio.loadmat(meanstd_file)
        test_mean = np.array(matfile['mean_test'])
        test_std = np.array(matfile['std_test'])
        # Save predictions to a matfile to open later in matlab
        mdict = {"Recovery": Recovery}
        # sio.savemat(savename, mdict
        io.imsave('Recovery.tif', np.float32(Recovery))
