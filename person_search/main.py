from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap, QPalette, QBrush

from ui import Ui_MainWindow # 导入ui文件转换后的py文件
from PyQt5.QtWidgets import QFileDialog, QLabel, QWidget

import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *
import cv2
from reid.data import make_data_loader
from reid.data.transforms import build_transforms
from reid.modeling import build_model
from reid.config import cfg as reidCfg

class mywindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def  __init__ (self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        #左侧文件及文件夹载入
        self.pushButton_query.clicked.connect(self.open_file_query)
        self.pushButton_ref.clicked.connect(self.open_file_ref)
        self.pushButton_open.clicked.connect(self.open_pic_query)
        #query图片显示
        self.label_img_query = QLabel(self.horizontalGroupBox_2)
        #右侧显示生成图片，label[0-20]
        self.label_1 = QLabel(self.verticalGroupBox_2)
        self.label_2 = QLabel(self.verticalGroupBox_2)
        self.label_3 = QLabel(self.verticalGroupBox_2)
        self.label_4 = QLabel(self.verticalGroupBox_2)
        self.label_5 = QLabel(self.verticalGroupBox_2)
        self.label_6 = QLabel(self.verticalGroupBox_2)
        self.label_7 = QLabel(self.verticalGroupBox_2)
        self.label_8 = QLabel(self.verticalGroupBox_2)
        self.label_9 = QLabel(self.verticalGroupBox_2)
        self.label_10 = QLabel(self.verticalGroupBox_2)
        self.label_11 = QLabel(self.verticalGroupBox_2)
        self.label_12 = QLabel(self.verticalGroupBox_2)
        self.label_13 = QLabel(self.verticalGroupBox_2)
        self.label_14 = QLabel(self.verticalGroupBox_2)
        self.label_15 = QLabel(self.verticalGroupBox_2)
        self.label_16 = QLabel(self.verticalGroupBox_2)
        self.label_17 = QLabel(self.verticalGroupBox_2)
        self.label_18 = QLabel(self.verticalGroupBox_2)
        self.label_19 = QLabel(self.verticalGroupBox_2)
        self.label_20 = QLabel(self.verticalGroupBox_2)

        #预测按钮绑定触发时间
        self.pushButton_start.clicked.connect(self.process)
        #img_load，生成list，方便后期循环显示
        self.labels = [self.label_1,self.label_2,self.label_3,self.label_4,self.label_5,self.label_6,self.label_7,self.label_8,self.label_9,self.label_10,
                       self.label_11,self.label_12,self.label_13,self.label_14,self.label_15,self.label_16,self.label_17,self.label_18,self.label_19,self.label_20,]

    def open_file_query(self):        # open query file
        foldername = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        print(foldername)
        self.lineEdit_query.setText(foldername)
        self.query = foldername
    def open_file_ref(self):           # open reference file
        foldername = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        print(foldername)
        self.lineEdit_ref.setText(foldername)
        self.reference = foldername
    def open_pic_query(self):          # open pic and show
        filename, filetype =QFileDialog.getOpenFileName(self, "选取文件", "./", "All Files(*);;Text Files(*.jpg)")
        print(filename, filetype)
        # self.lineEdit_open.setText(filename)
        self.label_img_query.setGeometry(86, 10, 130, 190) #设置label容器内图片的相对显示位置
        self.showImage = QPixmap(filename).scaled(self.label_img_query.width(), self.label_img_query.height()) #图片自适应label大小
        self.label_img_query.setPixmap(self.showImage)    #显示图片
        #judge the choose file is exists
        dst_dir = 'query/'
        if not os.path.isfile(filename):
            print("%s not exist!" % (filename))
        else:
            shutil.rmtree(r'query/')
            fpath, fname = os.path.split(filename)  # 分离文件名和路径
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)  # 创建路径
            shutil.copy(filename, dst_dir + fname)  # 复制文件
            print("copy %s -> %s" % (filename, dst_dir + fname))
            self.key_name_1 = fname.split('_')[0]     #通过标签提取检索id，方便后期正确率计算
            print('id:' + self.key_name_1)
            self.lineEdit_open.setText(dst_dir + fname)   #文本框显示

            # judge output file is exists,if not mkdir it
            output = 'output'
            if os.path.exists(output):
                shutil.rmtree(output)  # delete output folder
            os.makedirs(output)  # make new output folder
    #进行处理
    def process(self):

            with torch.no_grad():    #所有计算得出的tensor的requires_grad设置为False，《=====》禁止反向传播过程中自动求导
                parser = argparse.ArgumentParser()
                parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help="模型配置文件路径")
                parser.add_argument('--data', type=str, default='data/coco.data', help="数据集配置文件所在路径")
                parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='模型权重文件路径')
                # parser.add_argument('--images', type=str, default='data/samples', help='需要进行检测的图片文件夹')
                # parser.add_argument('-q', '--query', default=r'query', help='查询图片的读取路径.')
                parser.add_argument('--img-size', type=int, default=416, help='输入分辨率大小')
                parser.add_argument('--conf-thres', type=float, default=0.1, help='物体置信度阈值')
                parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS阈值')
                parser.add_argument('--dist_thres', type=float, default=1.0, help='行人图片距离阈值，小于这个距离，就认为是该行人')
                parser.add_argument('--fourcc', type=str, default='mp4v',
                                    help='fourcc output video codec (verify ffmpeg support)')
                parser.add_argument('--output', type=str, default='output', help='检测后的图片或视频保存的路径')
                parser.add_argument('--half', default=False, help='是否采用半精度FP16进行推理')
                parser.add_argument('--webcam', default=False, help='是否使用摄像头进行检测')
                opt = parser.parse_args()

                #参数设置
                cfg = 'cfg/yolov3.cfg'
                data = 'data/coco.data'
                weights = 'weights/yolov3.weights'
                #images = 'data/samples'
                output = 'output'
                fourcc = 'mp4v'
                img_size = 416
                # conf_thres = 0.5
                # nms_thres = 0.5
                dist_thres = 1.0            #行人图片距离阈值
                save_txt = False            #设置文本保存
                save_images = True          #开启生成图片保存
                images = self.reference     #设置需要进行检测的图片文件夹
                print(images)

                # Initialize
                device = torch_utils.select_device(force_cpu=False)
                torch.backends.cudnn.benchmark = False  # set False for reproducible results
                # if os.path.exists(output):
                #     shutil.rmtree(output)  # delete output folder
                #os.makedirs(output)  # make new output folder
                print('Initialize is ok')
                ############# 行人重识别模型初始化 #############
                query_loader, num_query = make_data_loader(reidCfg)
                reidModel = build_model(reidCfg, num_classes=1453)
                reidModel.load_param(reidCfg.TEST.WEIGHT)
                reidModel.to(device).eval()  #转换为测试模式

                query_feats = [] #待连接的张量序列，可以是任意相同Tensor类型的python 序列
                query_pids = []

                for i, batch in enumerate(query_loader):
                    print('batch query_loader is ok')
                    with torch.no_grad():
                        img, pid, camid = batch
                        img = img.to(device)
                        # print(img.size())    #img:torch.Size([1, 3, 256, 128])
                        feat = reidModel(img)  # 待查询图片，每张图片特征向量2048 torch.Size([1, 2048])
                        query_feats.append(feat)    #[tensor([[ 0.07704,...,0.00000]])]
                        query_pids.extend(np.asarray(pid))  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
                query_feats = torch.cat(query_feats, dim=0)  #torch.cat,对query_feat进行扩维，tensor([[ 0.07704,...,0.00000]])，torch.Size([NONE, 2048])
                print("The query feature is normalized")
                # 归一化的目的是将向量转换为单位向量，即其长度为1
                query_feats = torch.nn.functional.normalize(query_feats, dim=1, p=2)  # 计算出查询图片的特征向量,对二维张量进行挤压（归一化），p为范数
                ############# 行人检测模型初始化 #############
                model = Darknet(cfg, img_size)

                # Load weights   模型的权重加载到指定的模型中
                if weights.endswith('.pt'):  # pytorch format
                    model.load_state_dict(torch.load(weights, map_location=torch.device('cpu'))['model'])
                else:  # darknet format
                    _ = load_darknet_weights(model, weights)
                # Eval mode
                model.to(device).eval()
                # Half precision
                opt.half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA
                if opt.half:
                    model.half()

                # Set Dataloader
                vid_path, vid_writer = None, None
                if opt.webcam:
                    save_images = False

                dataloader = LoadImages(images, img_size=img_size, half=opt.half)   #path, img, img0, self.cap

                # Get classes and colors
                # parse_data_cfg(data)['names']:得到类别名称文件路径 names=data/coco.names
                classes = load_classes(parse_data_cfg(data)['names'])  # 得到类别名列表: ['person', 'bicycle'...]
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]  # 对于每种类别随机使用一种颜色画框
                # Run inference
                t0 = time.time()   #计算检测需要的时间
                pred_imgs = 0     #计算预测正确的图片数量
                # 保存检测过的图片
                plot_imgs = 'data/plot_imgs'    #绘制过的
                if os.path.exists(plot_imgs):
                    shutil.rmtree(plot_imgs)  # delete output folder
                os.makedirs(plot_imgs)  # make new output folder
                not_plot_imgs = 'data/not_plot_imgs'  #未绘制的
                if os.path.exists(not_plot_imgs):
                    shutil.rmtree(not_plot_imgs)  # delete output folder
                os.makedirs(not_plot_imgs)  # make new output folder

                for i, (path, img, im0, vid_cap) in enumerate(dataloader):      #img处理过的，im0原图
                    #print('enter for is ok')
                    t = time.time()
                    save_path = str(Path(output) / Path(path).name)  # 保存的路径       str_type
                    # Get detections shape: (3, 416, 224)
                    img = torch.from_numpy(img).unsqueeze(0).to(device)  # 对数组进行升维,torch.Size([1, 3, 416, 224])
                    pred, _ = model(img)  # 经过处理的网络预测，和原始的
                    #print(pred.float())
                    #print(type(pred.float()))
                    #with torch.no_grad():     #反向传播时,不会自动求导
                    det = non_max_suppression(pred.float())[0]  # torch.Size([1, 7])
                    save_plot_imgs = str(Path(plot_imgs) / Path(path).name)  # 保存的路径       str_type
                    save_not_plot_imgs = str(Path(not_plot_imgs) / Path(path).name)  # 保存的路径       str_type
                    if det is not None and len(det) > 0:
                        # 坐标转换，将预测信息（相对img_size 416*224）映射回原图 img0 size 128*64，四舍五入
                        # 对det的所有行以及前四列进行操作 （416, 224）--> （128, 64）
                        #print(img.shape)  # img.shape = ([1, 3, 416, 224]) bs,chw
                        #print(im0.shape)  # im0.shape = (128, 64, 3) hwc
                        # print(det[:, :4])  #前四列
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        print('%gx%g ' % img.shape[2:], end='')  # print image size '288x416'
                        for c in det[:, -1].unique():  # 对图片的所有类进行遍历循环
                            n = (det[:, -1] == c).sum()  # 得到了当前类别的个数，也可以用来统计数目
                            if classes[int(c)] == 'person':
                                print('%g %ss' % (n, classes[int(c)]), end=', ')  # 打印个数和类别

                        count = 0
                        gallery_img = []
                        gallery_loc = []
                        # (x1y1x2y2, obj_conf, class_conf, class_pred)
                        for *xyxy, conf, cls_conf, cls in det:  # 对于最后的预测框进行遍历
                            # *xyxy: 对于原图来说的左上角右下角坐标: [xmin,ymin,xmax,ymax]
                            if save_txt:  # Write to file
                                with open(save_path + '.txt', 'a') as file:
                                    file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                            # Add bbox to the image
                            label = '%s %.2f' % (classes[int(cls)], conf)  # 'person 1.00'
                            if classes[int(cls)] == 'person':
                                # plot_one_bo x(xyxy, im0, label=label, color=colors[int(cls)])
                                xmin = int(xyxy[0])
                                ymin = int(xyxy[1])
                                xmax = int(xyxy[2])
                                ymax = int(xyxy[3])
                                w = xmax - xmin
                                h = ymax - ymin
                                # 如果检测到的行人太小了，感觉意义也不大
                                # 根据实际情况进行设置
                                if w * h > 500:
                                    gallery_loc.append((xmin, ymin, xmax, ymax))
                                    crop_img = im0[ymin:ymax, xmin:xmax]  # HWC (602, 233, 3)
                                    crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                                    crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                                    gallery_img.append(crop_img)

                        if gallery_img:
                            gallery_img = torch.cat(gallery_img, dim=0)  #torch.Size([1, 3, 416, 224])
                            gallery_img = gallery_img.to(device)
                            gallery_feats = reidModel(gallery_img)  # torch.Size([8, 2048])
                            print("The gallery feature is normalized")
                            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量

                            # m: 1
                            # n: 1
                            m, n = query_feats.shape[0], gallery_feats.shape[0]
                            # pow实现逐行平方，sum实现保留维数的情况下元素逐行相加，expand实现维数扩展，\为续行符
                            # gallery与query对应
                            distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                                      torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                            # out=(beta∗M)+(alpha∗mat1@mat2)
                            # qf^2 + gf^2 - 2 * qf@gf.t()
                            # distmat - 2 * qf@gf.t()
                            # distmat: qf^2 + gf^2 #矩阵相乘
                            # qf: torch.Size([1, 2048])
                            # gf: torch.Size([8, 2048]))
                            distmat.addmm_(1, -2, query_feats, gallery_feats.t()) #计算查询特征和库特征之间的距离
                            # distmat = (qf - gf)^2
                            #print(distmat)
                            distmat = distmat.cpu().numpy()  # <class 'tuple'>: (3, 12)
                            distmat = distmat.sum(axis=0) / len(query_feats)  # 平均一下query中同一行人的多个结果
                            index = distmat.argmin()   #取最小值下标，判断是否相似
                            if distmat[index] < dist_thres:
                                print('距离：%s' % distmat[index])
                                try:
                                    cv2.imwrite(save_plot_imgs, im0)  #将绘制过的图片放入plot_imgs中
                                except:
                                    print('save plot imgs is error')
                                pred_imgs += 1  #成功预测图片自增一
                                plot_one_box_find(gallery_loc[index], im0, label='find!', color=colors[int(cls)])  #正确框，绿色

                            else:
                                cv2.imwrite(save_not_plot_imgs, im0) #将未绘制的图片放入not_plot_imgs中
                                plot_one_box_notfind(gallery_loc[index], im0, label='find!', color=colors[int(cls)]) #错误框，红色
                    print('Done. (%.3fs)' % (time.time() - t))  #打印预测所需时间

                    if opt.webcam:  # Show live webcam
                        cv2.imshow(weights, im0)

                    if save_images:  # Save image with detections
                        if dataloader.mode == 'images':
                            cv2.imwrite(save_path, im0)
                        else:
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer

                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps,
                                                             (width, height))
                            vid_writer.write(im0)
                    # print('start to copy file to output')
                if save_images:
                    save_file_path = os.getcwd() + os.sep + output   #保存路径
                    print('Results saved to %s' % save_file_path)
                    if platform == 'darwin':  # macos
                        os.system('open ' + output + ' ' + save_path)
                time_use = time.time() - t0
                print('Done. (%.3fs)' % time_use)  #用时
                time_used=format(time_use,'.3f')
                self.plainTextEdit.appendPlainText(f'time_use:{time_used}s')  #向plaintext中追加文本
                save_plot_path = os.getcwd() + os.sep + plot_imgs
                save_not_plot_path = os.getcwd() + os.sep + not_plot_imgs
                try:
                    X=54            #output 中1-1 X的初始坐标
                    Y=18            #1-1 Y的初始坐标
                    right_imgs=0    #通过标签提取id，对比query的id，计算正确率
                    all_imgs = 0    #计算refrence中的图片个数
                    count_X = 0     #控制列数
                    file_path = []  #保存output喜爱所有文件的标签，保存为list
                    tp = 0  #绘制的图片中正样本
                    fp = 0  #绘制的图片中负样本
                    tn = 0  #未绘制的图片中的正样本
                    fn = 0  #未绘制的图片中的负样本
                    for name in os.listdir(save_file_path):  #利用循环向list中追加文件绝对路径
                        # print(name)
                        all_imgs +=1
                        if self.key_name_1 == name.split('_')[0]:  #提取标签name中的关键id，与query的id作对比
                            right_imgs +=1
                        file_path.append(os.path.join(save_file_path,name))
                    for name in os.listdir(save_plot_path):    #正样本内
                        # print(name)
                        # all_imgs +=1
                        if self.key_name_1 == name.split('_')[0]:  #提取标签name中的关键id，与save_plot_path的id作对比
                            tp +=1  #预测为正,实际为正
                        else:
                            fp +=1  #预测为正,实际为负
                    for name in os.listdir(save_not_plot_path): #负样本内
                        if self.key_name_1 == name.split('_')[0]:  #提取标签name中的关键id，与save_not_plot_path的id作对比
                            fn +=1  #预测为负,实际为正
                        else:
                            tn +=1  #预测为负，实际为负
                    # print('save_not_plot_imgs is ok')
                    # 通过所有文件标签计算正确率，保留两位小数
                    #tp(true->true).fp(false->true).fn(true->false).tn(false->false)
                    #accuracy = (tp+tn)/(tp+fn+fp+tn)
                    #precision = tp/(tp+fp)
                    fn = right_imgs - tp
                    tn = (all_imgs - right_imgs) - fp
                    pre_positive = tp+tn
                    pre_negative = fn+tn
                    act_positive = tp+fn
                    act_negative = fp+tn
                    all_positive = tp+fp
                    tpr = format(float(tp)/float(act_positive ),'.2f')  #真正率
                    fpr = format(float(fp) / float(act_negative), '.2f') #假正率
                    fnr = format(float(fn) / float(act_positive), '.2f') #假正率
                    tnr = format(float(tn) / float(act_negative), '.2f')  # 假正率

                    accuracy_rate = format(float(pre_positive)/float(all_imgs),'.2f') #正确预测的正反例数 /总数
                    precision_rate = format(float(tp) / float(all_positive), '.2f')#预测出是正的里面有多少真正是正的,可理解为查准率
                    precision = tp/all_positive
                    recall_rate = format(float(tp) / float(act_positive), '.2f')    #在实际正样本中,分类器能预测出多少.与真正率相等.可理解为查全率
                    recall = tp/act_positive
                    f1_score_x = 2*(precision*recall)
                    f1_score_y = recall+precision
                    f1_score_rate = format(float(f1_score_x) / float(f1_score_y), '.2f') #精确率和召回率的调和值,更接近于两个数较小的那个,所以精确率和召回率接近时.F值最大
                    # 向结果框内追加文本
                    self.plainTextEdit.appendPlainText(f'Accuracy_rate:{accuracy_rate}')    #准确率 ，预测正确的正负样本总数占总数的多少
                    self.plainTextEdit.appendPlainText(f'Precision_rate:{precision_rate}')  #查全率 ，预测正里面有多少实际正
                    self.plainTextEdit.appendPlainText(f'recall_rate:{recall_rate}')        #查准率 ，实际正中有多少预测正
                    self.plainTextEdit.appendPlainText(f'f1_score_rate:{f1_score_rate}')    #准确率和召回率的加权平均，越大越好
                    # 正则匹配
                    # for name in glob.glob(save_file_path + key_name + '_[A-Za-z0-9]*.jpg'):
                    #     file_path.append(os.path.join(save_file_path, name))
                    #     print(key_name)
                    #     print(type(key_name))
                    # print(file_path)
                    for i in range(len(file_path)):  #循环显示图片
                        # print(file_path[i])
                        self.labels[i].setGeometry(X, Y, 91, 140)  #为第i个label设置图片
                        jpg = QPixmap(file_path[i]).scaled(self.labels[i].width(), self.labels[i].height())#自适应
                        self.labels[i].setPixmap(jpg)
                        X += 141        #同行相邻label的X差值
                        count_X += 1    #控制列数
                        if count_X == 4:    #修改显示错误
                            X -= 2
                        if count_X == 5:    #列数控制为五行
                            count_X -= 5
                            X = 54
                            Y += 152    #同列相邻label的Y差值

                except:
                    print('output is error')


if __name__=="__main__":
    import sys
    app=QtWidgets.QApplication(sys.argv)
    ui = mywindow()
    ui.show()
    sys.exit(app.exec_())