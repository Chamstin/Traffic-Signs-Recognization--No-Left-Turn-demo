#include<stdio.h>
#include<stdlib.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<malloc.h>
#include<Windows.h>
#include<math.h>

using namespace std;
using namespace cv;



//rgb转hsv 使用opencv自带过滤后只有一个通道
void RGB2HSV(double red, double green, double blue, double& hue, double& saturation, double& intensity)
{

	double r, g, b;
	double h, s, i;

	double sum;
	double minRGB, maxRGB;
	double theta;

	r = red / 255.0;
	g = green / 255.0;
	b = blue / 255.0;

	minRGB = ((r < g) ? (r) : (g));
	minRGB = (minRGB < b) ? (minRGB) : (b);

	maxRGB = ((r > g) ? (r) : (g));
	maxRGB = (maxRGB > b) ? (maxRGB) : (b);

	sum = r + g + b;
	i = sum / 3.0;

	if (i < 0.001 || maxRGB - minRGB < 0.001)
	{
		h = 0.0;
		s = 0.0;
	}
	else
	{
		s = 1.0 - 3.0 * minRGB / sum;
		theta = sqrt((r - g) * (r - g) + (r - b) * (g - b));
		theta = acos((r - g + r - b) * 0.5 / theta);
		if (b <= g)
			h = theta;
		else
			h = 2 * 3.1415926 - theta;
		if (s <= 0.01)
			h = 0;
	}

	hue = (int)(h * 180 / 3.1415926);
	saturation = (int)(s * 100);
	intensity = (int)(i * 100);
}


//红色滤波
Mat Redfilter(Mat image ) {
	Mat image_copy;   //clone函数创建新的图片 彩色
	image_copy = image.clone();   //clone函数创建新的图片 
	int x, y;
	double B = 0.0, G = 0.0, R = 0.0, H = 0.0, S = 0.0, V = 0.0;
	int width1 = image_copy.cols * 3;
	int height1 = image_copy.rows;
	for (x = 0; x < height1; x++)
	{
		uchar* data = image_copy.ptr<uchar>(x);//获取第i行的首地址
		for (y = 0; y < width1; y += 3)
		{
			B = data[y];
			G = data[y + 1];
			R = data[y + 2];
			RGB2HSV(R, G, B, H, S, V);
			if ((H >= 270 && H <= 360 || H >= 0 && H <= 20) && (S >= 15 && S <= 360) && (V > 0 && V < 360))
				data[y] = data[y + 1] = data[y + 2] = 255;
			else data[y] = data[y + 1] = data[y + 2] = 0;
		}
	}

	Mat hsv_rgb;
	cvtColor(image_copy, hsv_rgb, COLOR_BGR2GRAY);//灰度化
	hsv_rgb = hsv_rgb > 120;
	//namedWindow("hsv红色保留图像", 1);
	//imshow("hsv红色保留图像", hsv_rgb);
	//waitKey(0);
	return hsv_rgb;
}


//灰度反转 白色过滤
Mat converse(Mat image) {
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			//image.at<uchar>(i, j) = 255 - image.at<uchar>(i, j); //灰度反转
			uchar temp = image.at<uchar>(i, j);
			if (temp < 76)
			{

				image.at<uchar>(i, j) = 255;
			}
			else image.at<uchar>(i, j) = 0;
		}
	}

	return image;
}

//hu不变矩计算函数
vector<double> HuMoment(Mat img)
{
	//imshow("hu",img);
	//waitKey(0);
	int Width = img.cols;
	int Height = img.rows;
	int Step = img.step;
	uchar* p_data = (uchar*)img.data;
	double m00 = 0, m11 = 0, m20 = 0, m02 = 0, m30 = 0, m03 = 0, m12 = 0, m21 = 0; //中心矩 
	double x0 = 0, y0 = 0; //计算中心距时所使用的临时变量（x-x'）
	double u20 = 0, u02 = 0, u11 = 0, u30 = 0, u03 = 0, u12 = 0, u21 = 0;//规范化后的中心矩
	vector<double> M(7);//HU不变矩
	double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;//临时变量， 
	int Center_x = 0, Center_y = 0;//重心
	int i, j;            //循环变量

	//  获得图像的区域重心
	double s10 = 0, s01 = 0, s00 = 0;  //0阶矩和1阶矩  
	for (j = 0; j < Height; j++)//y
	{
		for (i = 0; i < Width; i++)//x
		{
			s10 += i * p_data[j *Step + i];
			s01 += j * p_data[j *Step + i];
			s00 += p_data[j * Step + i];
		}
	}
	Center_x = (int)(s10 / s00 + 0.5);
	Center_y = (int)(s01 / s00 + 0.5);
	//  计算二阶、三阶矩(中心矩)
	m00 = s00;
	for (j = 0; j < Height; j++)
	{
		for (i = 0; i < Width; i++)//x 
		{
			x0 = (i - Center_x);
			y0 = (j - Center_y);
			m11 += x0 * y0 * p_data[j * Step + i];
			m20 += x0 * x0 * p_data[j * Step + i];
			m02 += y0 * y0 * p_data[j * Step + i];
			m03 += y0 * y0 * y0 * p_data[j * Step + i];
			m30 += x0 * x0 * x0 * p_data[j * Step + i];
			m12 += x0 * y0 * y0 * p_data[j * Step + i];
			m21 += x0 * x0 * y0 * p_data[j * Step + i];
		}
	}

	// 计算规范化后的中心矩: mij/pow(m00,((i+j+2)/2)
	u20 = m20 / pow(m00, 2);
	u02 = m02 / pow(m00, 2);
	u11 = m11 / pow(m00, 2);
	u30 = m30 / pow(m00, 2.5);
	u03 = m03 / pow(m00, 2.5);
	u12 = m12 / pow(m00, 2.5);
	u21 = m21 / pow(m00, 2.5);

	// 计算中间变量
	t1 = (u20 - u02);
	t2 = (u30 + 3 * u12);
	t3 = (3 * u21 + u03); //使用论文中的公式
	t4 = (u30 + u12);
	t5 = (u21 + u03);

	// 计算不变矩 
	M[0] = u20 + u02;
	M[1] = t1 * t1 + 4 * u11 * u11;
	M[2] = t2 * t2 + t3 * t3;
	M[3] = t4 * t4 + t5 * t5;

	double beta1 = 0, beta2 = 0, beta3 = 0, beta4 = 0, beta5 = 0, e = 0;

	//计算改良不变矩
	beta1 = sqrt(M[1])/M[0];
	beta2 = sqrt(M[3]) / M[2];
	beta3=(M[2]*m00) / (M[0]*M[1]);
	beta4 = M[2] / (pow(M[0],3));
	beta5=M[3] / (pow(M[0], 3));
	
	//计算图像离心率e
	e = ((m20 - pow(m02, 2)) + 4 * pow(m11, 2)) / pow((m20+m02),2);

	vector<double> ju(6);
	ju[0] = beta1;
	ju[1] = beta2;
	ju[2] = beta3;
	ju[3] = beta4;
	ju[4] = beta5;
	ju[5] = e;

	return ju;
}

//归一化函数
double f(double x) {
	double y;
	if (x < 1)y = x;
	else
		if (x >= 1) y = (1 / x);
	return y;
}


//hu不变矩测度输出函数
vector<int> huCompare(vector<vector<double>> Rois_hu, vector<double> hu_std) {
	vector<double> hu_comp;
	vector<int> confirm;

	//定义相似性测度函数和归一化函数


	for (size_t i = 0; i < Rois_hu.size(); i++)
	{
		double powSum = 0;
		for (size_t j = 0; j < 6; j++)
		{
			double temp = Rois_hu[i][j] / hu_std[j];
			powSum += f(temp);
		}
		powSum=1-(powSum/7); //相似性测度 越接近0 越匹配 我这里有负数
		hu_comp.push_back(powSum);
	}

	//float yuzhi = 0;
	//yuzhi = (0.13334308957337293 + 0.0503999338332478126 + 0.056065618899397540 + 0.052648889390233933+ 0.033232505993414874+ 0.073810155749248874+0.054297270978664214) / 7;

	for (size_t i = 0; i < hu_comp.size(); i++)
	{
		if (hu_comp[i] < 0.74) confirm.push_back(i); //

	}
	return confirm; //返回确认图像索引
}


//将图像强制转换成指定大小 目前指定宽度400 按比例缩放至最佳检测大小 宽高不能差太多
Mat trans(Mat image,int flag) { //flag =0 400 flag =1 100
	if (flag == 0) {
		double w_times = 0, h = 0; //浮点解决小到大
		w_times = (image.cols / 400.0);
		h = image.rows / w_times;
		int _input_height = (int)h, _input_width = 400;
		Mat imgCrop;
		int minHW = min(image.rows, image.cols);
		float ratio = max(_input_height, _input_width) / (float)minHW;
		int newH = max(_input_height, (int)(image.rows * ratio));
		int newW = max(_input_width, (int)(image.cols * ratio));
		resize(image, image, Size(newW, newH));
	}
	else if (flag == 1) {
		double w_times = 0, h = 0; //浮点解决小到大
		w_times = (image.cols / 100.0);
		h = image.rows / w_times;
		int _input_height = (int)h, _input_width = 100;
		Mat imgCrop;
		int minHW = min(image.rows, image.cols);
		float ratio = max(_input_height, _input_width) / (float)minHW;
		int newH = max(_input_height, (int)(image.rows * ratio));
		int newW = max(_input_width, (int)(image.cols * ratio));
		resize(image, image, Size(newW, newH));
	}
	return image;
}

//orb算法 在不同图片寻找相同特征 修改后进行图片分类
bool orb(Mat std, Mat roi)
{

	//太小会导致关键点识别失败进行双线性插值到400*400调用上方等比例变换
	roi = trans(roi , 0);
	//imshow("ORB前处理", roi);
	//waitKey(0);


	//-- 初始化
	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); //蛮力汉民匹配器

	//-- 第一步:检测 Oriented FAST 角点位置
	detector->detect(std, keypoints_1);
	detector->detect(roi, keypoints_2);

	//-- 第二步:根据角点位置计算 BRIEF 描述子
	descriptor->compute(std, keypoints_1, descriptors_1);
	descriptor->compute(roi, keypoints_2, descriptors_2);

	Mat outStd;
	drawKeypoints(std, keypoints_1, outStd, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("ORB特征点", outStd);
	//waitKey(0);

	//-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 汉明 距离
	vector<DMatch> matches;

	if (keypoints_2.size() < keypoints_1.size())
		return FALSE;



	matcher->match(descriptors_1, descriptors_2, matches);

	//-- 第四步:匹配点对筛选
	double min_dist = 10000, max_dist = 0;

	//找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//cout << "最小距离：" << min_dist << endl;

	//当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.
	vector< DMatch > good_matches; 
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 30.0))
		{
			good_matches.push_back(matches[i]);
		}
	}


	int block1 = 0, block2 = 0, block3 = 0, block4 = 0;

	for (size_t t = 0; t < good_matches.size(); t++)
	{
		int x1 = 0, x2 = 0, y1 = 0, y2 = 0;
		x1 = keypoints_1[good_matches[t].queryIdx].pt.x;
		y1 = keypoints_1[good_matches[t].queryIdx].pt.y;
		x2 = keypoints_2[good_matches[t].trainIdx].pt.x;
		y2 = keypoints_2[good_matches[t].trainIdx].pt.y;

		if (x1 < 200 && y1 < 200 && x2 < 200 && y2 < 200)
		{
			block1++; //主特征 后证伪
		}
		if (x1 < 200 && y1 > 200 && x1 < 200 && y1 > 200)
		{
			block2++;// 后证主特征
		}
		if (x1 > 200 && y1 > 200 && x1 > 200 && y1 > 200)
		{
			block3++; //主特征
		}
		if (x1 > 200 && y1 < 200 && x1 > 200 && y1 < 200)
		{
			block4++; //负特征
		}
	}

	if (block1 > 5 && block2 > 5 && block3 > 6 && block4 > 25)
		return 1;
	else
		return 0;


}

//目标框出
void draw_rec(Mat img,int r,int x,int y) {
	int r_height = r*2;
	int px = x - r;
	int py = y - r;
	cv::Point point(px, py);
	Rect rect(px, py, r_height + 10, r_height + 10);
	rectangle(img, rect,(0,0,255),2);
	putText(img,"MARK:1.No Left Turn",point, FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1);//字号和值成正比非office
	//namedWindow("【画图预览】", 0); //改变窗口不能放大的限制
	//imshow("【画图预览】", img);//显⽰处理后的图像
	//waitKey(0);
}



int main() {


/*图像预处理功能块*/
	Mat image = imread("./imageforge/t20.png");
	//imshow("【图片显示，按键继续】", image);
	//waitKey(0);
	image = trans(image,0);

	Mat img_dst = image.clone(); //最终的画布 最后使用非预处理功能

	//imshow("【大小调整显示，按键继续】", image);
	//waitKey(0);

	Mat img_gray;
	cvtColor(image, img_gray, COLOR_BGR2GRAY); //后面使用的灰度图像 剪裁使用

	//以下开始进行hsv色彩空间转换以及 颜色过滤
	 
	Mat hsv_rgb = Redfilter(image);

	
	

	//中值滤波 
	Mat midBlur_red_rgb;
	medianBlur(hsv_rgb, midBlur_red_rgb, 3); 
	//namedWindow("【midBlur_red_hsv显示，按键继续】", 1); //改变窗口不能放大的限制
	//imshow("【midBlur_red_hsv显示，按键继续】", midBlur_red_rgb);
	//waitKey(0);




/*标志区域定位与提取*/
	//哈夫圆形检测
	Mat img_copy2;
	img_copy2 = image.clone(); //哈夫圆检测绘制画板图形

	vector<Vec3f> circles;
	int maxR = min(midBlur_red_rgb.rows,midBlur_red_rgb.cols);//最大检测半径
	Canny(midBlur_red_rgb, midBlur_red_rgb, 50, 100);

	//imshow("【canny】", midBlur_red_rgb);//显⽰处理后的图像
	//waitKey(0);

	HoughCircles(midBlur_red_rgb, circles, HOUGH_GRADIENT, 1, 10, 100, 55, 3, maxR);  //哈夫圆检测没有检测到圆 与没有canny边缘提取相关？ 55
	Scalar circleColor = Scalar(255, 0, 0);//圆形的边缘颜⾊
	Scalar centerColor = Scalar(0, 0, 255);//圆⼼的颜⾊
	for (int i = 0; i < circles.size(); i++) {
		Vec3f c = circles[i]; //哈夫变换输出 圆心和半径
		circle(img_copy2, Point(c[0], c[1]), c[2], circleColor, 2, LINE_AA);//画圆
		circle(img_copy2, Point(c[0], c[1]), 2, centerColor, 2, LINE_AA);//圆⼼
	}
	//namedWindow("【哈夫圆检测】", 1); //改变窗口不能放大的限制
	//imshow("【哈夫圆检测】", img_copy2);//显⽰处理后的图像
	//waitKey(0);
	//circle存储的是两个圆 为三维数组 通道1：圆心x, 通道2：圆心y 通道3：圆的半径


	vector<Mat> Rois;//创建剪裁数组

	if (circles.size() > 0) //判断哈夫变换有没有检测到圆  //circles.size() > 0
	{
		for (size_t i = 0; i < circles.size(); i++)
		{
			Vec3f c1 = circles[i];
			int pX = c1[0] - c1[2];
			int pY = c1[1] - c1[2];
			int r_width = c1[2]*2;
			int r_height = c1[2] * 2;
			Rect rect(pX,pY,r_width+5,r_height+5);
			Mat roi = img_gray(rect);
			Rois.push_back(roi); //存储剪裁 roi里面都是 灰度原图模板匹配时用
		}
	}


	//Rois.push_back(img_gray); //存储std时使用 平常废弃

	//for (size_t i = 0; i < Rois.size(); i++)
	//{
	//	namedWindow("【剪裁预览】,i", 0); //改变窗口不能放大的限制
	//	imshow("【剪裁预览】,i", Rois[i]);//显⽰处理后的图像
	//	waitKey(0);
	//}

/*标志识别和标志纹理提取功能块（其中标志纹理提取封装成为函数合并与此功能块中）*/


	/*第一纹理识别：hu不变矩计算匹配*/
	vector<vector<double>> Rois_hu;

	for (size_t i = 0; i < Rois.size(); i++) //计算每个剪裁片的不变矩 传入canny后的图计算
	{
		Mat temp;
		Canny(Rois[i], temp, 50, 100);
		Rois_hu.push_back(HuMoment(temp)); //传入的是边缘特征不需要传入颜色反转的进去
	}

	vector<double> hu_std(6);//标准不变矩 hu不变矩在图像平移、旋转和比例变化时保持不变。不用考虑剪裁图像大小
	vector<int> confirm;

	//计算并保存标准(std_feature0.jpg)hu不变矩 利用std文件算出 并写入
	hu_std[0]= 0.090578238632476202;
	hu_std[1] = 2288.3209578146093;
	hu_std[2] = 749302.64995777595;
	hu_std[3] = 0.0067209931665668269;
	hu_std[4] = 0.0030584991498335667;
	hu_std[5] = -0.25108512282973472;

	confirm = huCompare(Rois_hu, hu_std); 


	//for (size_t i = 0; i < confirm.size(); i++)
	//{
	//	//namedWindow("【confirm预览】,i", 0); //改变窗口不能放大的限制
	//	//imshow("【confirm预览】,i", Rois[confirm[i]]);//显⽰处理后的图像
	//	//waitKey(0);
	//}

	/*第二纹理识别：orb特征匹配*/

	//进行confirm剪裁 ORB匹配
	vector<bool> ORB_confirm(confirm.size());
	Mat std = imread("./imageforge/std_feature0.png");
	//Mat roi = imread("./imageforge/std_feature0.png"); //测试用
	//ORB_confirm[0] = orb(std, roi); //测试用
	for (size_t i = 0; i < confirm.size(); i++)
	{
		ORB_confirm[i] = orb(std,converse(Rois[confirm[i]])); //保存确认索引
	}



	//for (size_t i = 0; i < confirm.size(); i++)
	//{
	//	if (ORB_confirm[i]) {
	//		namedWindow("【最终预览】,i", 0); //改变窗口不能放大的限制
	//		imshow("【最终预览】,i", Rois[confirm[i]]);//显⽰处理后的图像
	//		waitKey(0);
	//		cout << "pasuse" << endl;
	//	 }
	//}


/*目标框出功能块*/
	//绘制目标识别区域
	int count_f = 0;
	for (size_t i = 0; i < ORB_confirm.size(); i++)
	{
		int idx = i;
		if (ORB_confirm[i]) {
			int x = circles[idx].val[0];
			int y = circles[idx].val[1];
			int r = circles[idx].val[2];
			draw_rec(img_dst,r,x,y);
			namedWindow("【目标框出预览,按回车继续】", 1); //改变窗口不能放大的限制
			imshow("【目标框出预览,按回车继续】", img_dst);//显⽰处理后的图像
			waitKey(0);
			count_f++;
		}
	}

	if (count_f == 0) {
		
		printf("\n 未检测到目标, 程序退出 \n");
		return 0;
	}

	imwrite("./imgOut/检测图完成图片.jpg", img_dst);

	cout << "\n" << "程序结束，文件已保存在imgOut/检测图完成图片.jpg" <<"\n" << endl;

	return 0;	
}
