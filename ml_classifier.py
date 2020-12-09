import os
import csv
import random
from PIL import Image,ImageFilter
from sklearn import svm,tree,ensemble,model_selection
import joblib
import numpy as np
import matplotlib.pyplot as plt
base_directory=r'D:\PycharmProjects\pythonProject'

def fun01(y):
    return y*2.0
def fun02(y):
    return y*0.5

def load_resource():
    print('load_resource')
    i=0
    train_images=[]
    train_dir=os.path.join(base_directory,r'jiangnan2020\train\train')
    train_labels=[]
    labels=[]
    i=0
    with open(os.path.join(base_directory,r'jiangnan2020\train.csv'),'r') as file:
        csv_reader=csv.reader(file)
        csv_list=list(csv_reader)
        csv_list.pop(0)
        for row in csv_list:
            if i >10000: break
            i += 1
            base_name=row[0]+'.jpg'
            train_images.append(os.path.join(train_dir,base_name))
            labels.append(int(row[1]))
    # image_contents = []
    image_contents_transfer=[]
    for path in train_images:
        with Image.open(path) as im:
            # data=[im.getpixel((w,h)) for w in range(200) for h in range(200)]
            # np.set_printoptions(threshold=np.inf)
            # im_array=np.array(im).reshape(1, -1)[0]
            # image_contents.append(im_array)

            # random_gau=random.randint(3,9)*0.5
            image_contents_transfer.append(np.array(im).reshape(1, -1)[0])


            # image_contents_transfer.append(np.array(Image.eval(im,fun01)).reshape(1, -1)[0])
            # image_contents_transfer.append(np.array(Image.eval(im, fun02)).reshape(1, -1)[0])
            #
            # temp=random.randint(0,2)
            # if temp==0:image_contents_transfer.append(np.array(im.transpose(Image.FLIP_LEFT_RIGHT)).reshape(1,-1)[0])
            # else:image_contents_transfer.append(np.array(im.transpose(Image.FLIP_TOP_BOTTOM)).reshape(1, -1)[0])
            # temp=random.randint(1,4)*90
            # image_contents_transfer.append(np.array(im.rotate(temp)).reshape(1,-1)[0])
    return image_contents_transfer,labels

def get_predict(classifier,directory=os.path.join(base_directory,r'jiangnan2020\test\test')):
    print('predict')
    row_list=[]
    i=0
    with open(os.path.join(base_directory,r'jiangnan2020\test.csv'),'r') as f:
        csv_reader=csv.reader(f)
        row_list_reader=list(csv_reader)
        row_list_reader.pop(0)
        for row_reader in row_list_reader:
            basename=row_reader[0]+'.jpg'
            with Image.open(os.path.join(directory,basename)) as img:
                # if i>300:break
                # i+=1
                result = classifier.predict(np.array(img).reshape(1,-1))
                row=[]
                file=row_reader[0]
                row.append(file)
                row.append(result[0])
                row_list.append(row)
    print(row_list)
    with open('rfc3.csv','w',encoding='utf-8',newline='') as file:
        csv_writer=csv.writer(file)
        csv_writer.writerow(['id','label'])
        csv_writer.writerows(row_list)


def train_rfc(images,labels):
    print('train_rfc')
    # rfc = ensemble.RandomForestClassifier(n_estimators=80, random_state=17,oob_score=True)此参数准确率为83，输入只使用原始图片。
    rfc=ensemble.RandomForestClassifier(n_estimators=80,random_state=17,max_features=37)
    # rfc.fit(images[:30000],labels[:30000])
    # rfc.fit(images[30000:],labels[30000:])
    rfc.fit(images,labels)
    joblib.dump(rfc,'rfc3.pkl')
    return rfc

def searchCV(images,labels):
    print("searchcv")

    # param_nestimator={'n_estimators':range(100,200,10)}
    # rfs=ensemble.RandomForestClassifier(max_features=29,random_state=17)
    # gsearch_estimator=model_selection.GridSearchCV(rfs,param_grid=param_nestimator,n_jobs=3,scoring='roc_auc',cv=5)
    # gsearch_estimator.fit(images,labels)
    # print(gsearch_estimator.best_params_,gsearch_estimator.cv_results_)

   # 取上面的最佳n_estimator继续搜索最佳参数
   #  param_max = {'max_features':range(29,40,2)}
   #  rfs=ensemble.RandomForestClassifier(n_estimators=100,random_state=10)
   #  gsearch_estimator=model_selection.GridSearchCV(rfs,param_grid=param_max,n_jobs=1,scoring='roc_auc',iid=False,cv=5)
   #  gsearch_estimator.fit(images,labels)
   #  print(gsearch_estimator.best_params_,gsearch_estimator.cv_results_)

def test_predict(classifier,directory=os.path.join(base_directory,r'jiangnan2020\train\train')):
    print('test_predict')
    i=0
    count=0
    with open(os.path.join(base_directory,r'jiangnan2020\train.csv'),'r') as f:
        csv_reader=csv.reader(f)
        row_list_reader=list(csv_reader)
        row_list_reader.pop(0)
        for row_reader in row_list_reader:
            i=i+1
            if i<15001:continue
            if i>18000:break
            basename=row_reader[0]+'.jpg'
            with Image.open(os.path.join(directory,basename)) as img:
                result = classifier.predict(np.array(img).reshape(1,-1))
                if result[0]!=int(row_reader[1]):
                    print(basename)
                    print(result[0],row_reader[1])
                else:
                    count=count+1
    print(count)
    print('count: ',count/(18000-15001))






if __name__ == "__main__":
    print('----------------------------')
    temp=load_resource()
    # searchCV(temp[0],temp[1])
    # rfc=joblib.load('rfc1.pkl')
    # rfc=train_rfc(temp[0],temp[1])
    # get_predict(rfc)
    # test_predict(rfc)
    # validate_images(temp[0],temp[1],temp[2],temp[3],temp[4])




def validate_images(images,labels,images_convert,image_transfer,transfer_labels):
    print('confirm_image')
    scores_list = []
    scores_list_transfer=[]
    source_list_convert=[]
    for i in range(20, 23):
        rfc = ensemble.RandomForestClassifier(n_estimators=i, random_state=23)
        rfc_s = model_selection.cross_val_score(rfc, images, labels, cv=10, scoring='accuracy')
        scores_list.append(rfc_s.mean())

        rfc = ensemble.RandomForestClassifier(n_estimators=25, random_state=i)
        temp3 = model_selection.cross_val_score(rfc, image_transfer, transfer_labels, cv=10, scoring='accuracy')
        scores_list_transfer.append(temp3.mean())

        rfc = ensemble.RandomForestClassifier(n_estimators=25, random_state=i)
        temp1 = model_selection.cross_val_score(rfc, images_convert, labels, cv=10, scoring='accuracy')
        source_list_convert.append(temp1.mean())

    plt.plot(range(20, 23), scores_list, label='rfc')
    plt.plot(range(20, 23), scores_list_transfer, label='transfer')
    plt.plot(range(20, 23), source_list_convert, label='convert')
    plt.legend()
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.show()
    return

def validate_rfc(images,labels):
    print('confirm')
    scores_list=[]
    for i in range(20,30):
        rfc = ensemble.RandomForestClassifier(n_estimators=25,random_state=i)
        rfc_s = model_selection.cross_val_score(rfc, images,labels,cv=10,scoring='accuracy')
        scores_list.append(rfc_s.mean())
    plt.plot(range(10,20),scores_list,label='rfc')
    plt.legend()
    plt.xlabel('random_state')
    plt.ylabel('accuracy')
    plt.show()
    print('最优值:',max(scores_list))
    print('最优random_state为',scores_list.index(max(scores_list))+10)
    return rfc

def train_svm(train_images,train_labels):
    print('train')
    svcClassifier=svm.SVC(C=10000,gamma=0.001,tol=0.01)
    svcClassifier.fit(train_images,train_labels)
    joblib.dump(svcClassifier,'svm.pkl')
    return svcClassifier

def train_tree(images,labels):
    print('train_decisonTree')
    clf=tree.DecisionTreeClassifier()
    clf=clf.fit(images,labels)
    joblib.dump(clf, 'tree.pkl')
    return clf


