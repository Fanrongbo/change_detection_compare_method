from sklearn.metrics import auc,roc_curve,accuracy_score,cohen_kappa_score,classification_report,confusion_matrix
import numpy as np
from sklearn.preprocessing import minmax_scale 
import numpy.matlib
import matplotlib.pyplot  as plt
import cv2
import matplotlib as mpl
#The function aim to achieve evaluate the change detection preformance. All of pixels is vaild data.
class Metrics_2class:
    def __init__(self,predict,label):
        self.predict=predict
        self.label=label
    def metric(self,img=None, chg_ref=None):
        shape = img.shape
        chg_ref = np.array(chg_ref, dtype=np.float32)
        chg_ref = chg_ref / np.max(chg_ref)#normalization
        confusion_map = np.zeros((shape[0] * shape[1]))#confusion matrix
        img = np.reshape(img, [-1])
        chg_ref = np.reshape(chg_ref, [-1])
        # change_label = np.where(chg_ref == 1)[0]#location of change label
        # TP = (np.sum(img[change_label] == 1))#change right (tp)
        # FN=np.sum(img[change_label] == 0)#change false (FN)
        # unchange_label = np.where(chg_ref == 0)[0]#location of unchange label
        # TN = np.sum(img[unchange_label] == 0)#unchange right (TN)
        # FP=np.sum(img[unchange_label] == 1)#unchange false (FP) 
        # print(TN,FP,FN,TP)
        # print(TN,FP,FN,TP,change_label.shape,unchange_label.shape)
        lastRef=chg_ref.copy()
        lastRef=lastRef.ravel()
        test=img.copy()
        test=test.ravel()
        #|                  |PREDICTED VALUE |         |
        #|                  |POSITIVE|NEGTIVE|TOTAL NUM|
        #|OBSERVED|POSITIVE |   TP   |  FN   |    P    |
        #| VALUE  |NEGTIVE  |   FP   |  TN   |    N    |
        # print(type(TN))
        TN,FP,FN,TP = confusion_matrix(lastRef, test).ravel()
        # print(TN,FP,FN,TP)
        
        return TN,FN,FP,TP
    def draw_confusion(self):
        shape=self.predict.shape
        detection=(self.predict*255).reshape(shape[0],shape[1])
        ref=(self.label*255).reshape(shape[0],shape[1])
        
        (h,w)=detection.shape
        out=np.zeros((h,w,3))
        TP_color=(0,255,0)#BGR
        TN_color=(255,255,0)
        FN_color=(0,0,255)
        FP_color=(0,255,255)
        # print(out.shape,detection.shape)
        
        for i in range(0,shape[0]):
            for j in range(0,shape[1]): 
                if detection[i,j]==255 and ref[i,j]==255:#TP
                    out[i,j]=TP_color
                if detection[i,j]==0 and ref[i,j]==0:#TN
                    out[i,j]=TN_color
                if detection[i,j]==255 and ref[i,j]==0:#FN
                    out[i,j]=FN_color
                if detection[i,j]==0 and ref[i,j]==255:#FP
                    out[i,j]=FP_color
        cv2.imwrite('out_2class.png',out)
        print('The confusion map have been saved!')
        # plt.figure('confusion map')
        # fig, ax = plt.subplots() 
        # plt.title('confusion map')
        # labels = ['TP', 'TN', 'FN', 'FP']
        # colors = [TP_color,TN_color,FN_color,FP_color]
        # out = cv2.cvtColor(out.astype(np.uint8),cv2.COLOR_RGB2BGR)
        # plt.imshow(out)
        # plt.show()
    def evaluation(self):
        TN0,FN0,FP0,TP0=self.metric(1-self.predict,self.label)
        TN1,FN1,FP1,TP1=self.metric(self.predict,self.label)
        #Judge the correct result
        if TN1+TP1>TN0+TP0:
            TN,FN,FP,TP =TN1,FN1,FP1,TP1
        else:
            TN,FN,FP,TP =TN0,FN0,FP0,TP0
            self.predict=1-self.predict
        #Calculate the criteria
        change_label=TP+FN
        unchange_label=TN+FP
        all=TN+TP+FN+FP
        # P=TP+FN
        # N=FP+TN
        OA=(TP+TN)/all
        pre=((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/(np.square(all))
        k=(OA-pre)/(1-pre)
        recall=TP/(TP+FN)
        acc_chg = np.divide(float(TP), change_label)#accuracy of change
        acc_un = np.divide(float(TN), unchange_label)#accuracy of unchange
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%2class Criteria%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Total sample:',(change_label+unchange_label),'| Change Sample:',change_label,'| Unchange Sample:',unchange_label)
        print('Original image shape:',self.label.shape)
        print('TN:',TN,'| FP:',FP,'| FN:',FN,'| TP:',TP)
        print('Accuracy of Unchanged Regions is: %.4f' % (acc_un))
        print('Accuracy of Changed Regions is:   %.4f' % (acc_chg))
        print('Overall Accuracy:%.4f | Kappa:%.4f | Precision:%.4f | Recall:%.4f'%(OA,k,pre,recall))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        plt.figure('ROC curve')
        N=self.label.shape[0]*self.label.shape[1]
        lastRef=(self.label.reshape(N,1)).copy()
        test=(self.predict.reshape(N,1)).copy()
        fpr, tpr, thresholds = roc_curve(lastRef, test )
        auc_value=auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % auc_value)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.savefig('ROC_curve.png')
        plt.clf()
        self.draw_confusion()#draw the confusion map


#The function aim to achieve evaluate the change detection preformance. Some of pixels is vaild data, not all of image pixels.
class Metrics_3class:
    def __init__(self,predict,label):
        self.predict=predict
        self.label=label
    def metric(self,predict,label):
        # predict=self.predict
        # label=self.label
        shape=predict.shape
        confusion_map=np.zeros((predict.shape[0]*predict.shape[1]))

        if label.shape!=predict.shape:
            print('shape must match!')
        if len(predict.shape)!=2 or len(label.shape)!=2:
            print('Input must be 2-Dimension')
        rows,cols=predict.shape
        N=rows*cols#Total original image pixels number
        TI=(predict.reshape(N,1)).copy()
        lastRef=(label.reshape(N,1)).copy()
        for i in range(0,N):
            if lastRef[i]!=0:#FP=0;TP=1;TN=2,FN=3
                
                # if TI[i]==1 and lastRef[i]==1:
                    # confusion_map[i]=1  #TP
                # if TI[i]==0 and lastRef[i]==2:
                    # confusion_map[i] = 2  # TN
                # if TI[i]==0 and lastRef[i]==1:
                    # confusion_map[i] = 3  # FN
                # if TI[i] == 1 and lastRef[i] == 2:
                    # confusion_map[i] = 4  # FP
                TI[i]=TI[i]+2 #Select the rigion of sample

        lastRef=lastRef[lastRef!=0]#Area of samples
        test=TI[(TI!=0) & (TI!=1)]
        TotalSample=np.sum(lastRef != 0)#Total sample pixels number
        lastRef=lastRef-1
        test=test-2

        #|                  |PREDICTED VALUE |         |
        #|                  |POSITIVE|NEGTIVE|TOTAL NUM|
        #|OBSERVED|POSITIVE |   TP   |  FN   |    P    |
        #| VALUE  |NEGTIVE  |   FP   |  TN   |    N    |
        # change_label = np.where(lastRef == 1)[0]#location of change label
        # TP = (np.sum(test[change_label] == 1))#change right (tp)
        # FN=np.sum(test[change_label] == 0)#change false (FN)
        # unchange_label = np.where(lastRef == 0)[0]#location of unchange label
        # TN = np.sum(test[unchange_label] == 0)#unchange right (TN)
        # FP=np.sum(test[unchange_label] == 1)#unchange false (FP) 
        TN,FP,FN,TP = confusion_matrix(lastRef, test).ravel()

        return TN,FN,FP,TP,lastRef,test
        
    def draw_confusion(self):
        shape=self.predict.shape
        detection=(self.predict*255).reshape(shape[0],shape[1])
        ref=(self.label).reshape(shape[0],shape[1])
        
        (h,w)=detection.shape
        out=np.zeros((h,w,3))
        TP_color=(0,255,0)#BGR
        TN_color=(255,255,0)
        FN_color=(0,0,255)
        FP_color=(0,255,255)
        Nosmaple=(255,0,0)
        
        for i in range(0,h):
            for j in range(0,w):
                if ref[i,j]==0:#nosample
                    out[i,j]=Nosmaple
                if detection[i,j]==255 and ref[i,j]==2:#TP
                    out[i,j]=TP_color
                if detection[i,j]==0 and ref[i,j]==1:#TN
                    out[i,j]=TN_color
                if detection[i,j]==255 and ref[i,j]==1:#FN
                    out[i,j]=FN_color
                if detection[i,j]==0 and ref[i,j]==2:#FP
                    out[i,j]=FP_color       
        cv2.imwrite('out_3class.png',out)
        print('The confusion map have been saved!')
        # plt.figure('confusion map')
        # fig, ax = plt.subplots() 
        # plt.title('confusion map')
        # labels = ['TP', 'TN', 'FN', 'FP']
        # colors = [TP_color,TN_color,FN_color,FP_color]
        # out = cv2.cvtColor(out.astype(np.uint8),cv2.COLOR_RGB2BGR)
        # plt.imshow(out)
        # plt.show()
    def evaluation(self):
        TN0,FN0,FP0,TP0,lastRef0,test0=self.metric(1-self.predict,self.label)
        TN1,FN1,FP1,TP1,lastRef1,test1=self.metric(self.predict,self.label)
        if TN1+TP1>TN0+TP0:
            TN,FN,FP,TP,lastRef,test =TN1,FN1,FP1,TP1,lastRef1,test1
        else:
            TN,FN,FP,TP,lastRef,test =TN0,FN0,FP0,TP0,lastRef0,test0
            self.predict=1-self.predict
        del TN0,FN0,FP0,TP0,lastRef0,test0,TN1,FN1,FP1,TP1,lastRef1,test1
        change_label=TP+FN
        unchange_label=TN+FP
        all=TN+TP+FN+FP
        # P=TP+FN
        # N=FP+TN
        OA=(TP+TN)/all
        pre=((TP+FP)*(TP+FN)+(FN+TN)*(FP+TN))/(np.square(all))
        k=(OA-pre)/(1-pre)
        recall=TP/(TP+FN)
        acc_chg = np.divide(float(TP), change_label)#accuracy of change
        acc_un = np.divide(float(TN), unchange_label)#accuracy of unchange
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%3class Criteria%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Original image shape:',self.label.shape)
        print('Total sample:',(change_label+unchange_label),'| Change Sample:',change_label,'| Unchange Sample:',unchange_label)
        print('TN:',TN,'| FP:',FP,'| FN:',FN,'| TP:',TP)
        print('Accuracy of Unchanged Regions is: %.4f' % (acc_un))
        print('Accuracy of Changed Regions is:   %.4f' % (acc_chg))
        print('Overall Accuracy:%.4f | Kappa:%.4f | Precision:%.4f | Recall:%.4f'%(OA,k,pre,recall))
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        plt.figure('ROC curve')
        N=self.label.shape[0]*self.label.shape[1]
        # lastRef=(self.label.reshape(N,1)).copy()
        # test=(self.predict.reshape(N,1)).copy()
        fpr, tpr, thresholds = roc_curve( lastRef,test )
        auc_value=auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % auc_value)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.savefig('ROC_curve_3class.png')
        plt.clf()
        self.draw_confusion()#draw the confusion map

def main_2class():
    ChangeRI=cv2.imread('river_ref.bmp',0)//255
    change_map=cv2.imread('ufmad_river.png',0)//255
    img_width, img_height=ChangeRI.shape
    AllRef = ChangeRI
    img_shape=(img_width, img_height)
    print('Calculating the evaluation criteria, please waiting...')
    metrics = Metrics_2class(change_map,AllRef)
    metrics.evaluation()
    
def main_3class():
    UnchangeRI=cv2.imread('barbara_unchange_map.png',0)
    ChangeRI=cv2.imread('barbara_change_map.png',0)
    change_map=cv2.imread('ufmad_barbara.png',0)//255
    img_width, img_height=ChangeRI.shape
    #数值有三种：0,1,2
    AllRef =  np.zeros((img_width, img_height))
    for i in range(0,img_width):
        for j in range(0,img_height):
            if UnchangeRI[i,j]==255:
                AllRef[i,j]=1
            if ChangeRI[i,j]==255:
                AllRef[i,j]=2
    print('Calculating the evaluation criteria, please waiting...')
    metrics = Metrics_3class(change_map,AllRef)
    metrics.evaluation()
if __name__ == '__main__':
    main_2class()
    main_3class()