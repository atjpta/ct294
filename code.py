
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
##########################################

# doc du lieu
dulieu = pd.read_csv("music_genre.csv")
##########################################
# so luong phan tu la 50005
len(dulieu)

# #loại bỏ các hàng lỗi
# dulieu.tempo.replace(to_replace = '?', value =np.nan, inplace=True)
# dulieu.duration_ms.replace(to_replace = -1, value =np.nan, inplace=True)
dulieu = dulieu.dropna()


# liet ke cac so luong va gia tri khac nhau cua bien
# Electronic     5000
# Anime          5000
# Jazz           5000
# Alternative    5000
# Country        5000
# Rap            5000
# Blues          5000
# Rock           5000
# Classical      5000
# Hip-Hop        5000
# Name: music_genre, dtype: int64
dulieu.music_genre.value_counts()
dulieu.describe()

# tất cả các thuộc tính
# 0 instance_id	
# 1 artist_name	
# 2 track_name	
# 3 popularity	
# 4 acousticness	
# 5 danceability	
# 6 duration_ms	
# 7 energy	
# 8 instrumentalness	
# 9 key	
# 10 liveness	
# 11 loudness	
# 12 mode	
# 13 speechiness	
# 14 tempo	
# 15 obtained_date	
# 16 valence	
# 17 music_genre

# loại bỏ các cột không cần thiết
x = dulieu.iloc[:,[3,4,5,7,8,10,11,13,16]]
y = dulieu.music_genre


##############################################

# phan chia du lieu thanh test va train
# train = 40000 phan tu
# test = 10000 phan tu
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=351)


###############################################

# xay dung mo hinh knn
# cho neighbors = 1000
knn = KNeighborsClassifier(100)
knn.fit(X_train, Y_train)

# xay dung mo hinh bayes dua tren phan phoi xac xuat tuan theo Gausian
bayes = GaussianNB()
bayes.fit(X_train, Y_train)

#xay dung mo hình cây quyết định
tree = DecisionTreeClassifier(random_state=351)
tree.fit(X_train, Y_train)

#############################################

#đánh giá mô hình

# knn
Y_pred_KNN = knn.predict(X_test)
print("\n==============================================================================\n")
print("knn có độ chính xác tổng thể là là: ", accuracy_score(Y_test, Y_pred_KNN)*100)
print("độ chính xác cho từng phân lớp :")
print(confusion_matrix(Y_test, Y_pred_KNN))

# bayes
Y_pred_bayes = bayes.predict(X_test)
print("\n==============================================================================\n")
print("bayes có độ chính xác tổng thể là là: ", accuracy_score(Y_test, Y_pred_bayes)*100)
print("độ chính xác cho từng phân lớp :")
print(confusion_matrix(Y_test, Y_pred_bayes))

# tree
Y_pred_tree = tree.predict(X_test)
print("\n==============================================================================\n")
print("tree có độ chính xác tổng thể là là: ", accuracy_score(Y_test, Y_pred_tree)*100)
print("độ chính xác cho từng phân lớp :")
print(confusion_matrix(Y_test, Y_pred_tree))

####################################
# giai thuat hold-out
print(cross_val_score(knn, x, y).mean()*100)
print(cross_val_score(bayes, x, y).mean()*100)
print(cross_val_score(tree, x, y).mean()*100)

####################################
#dự đoán 1 phần tử mới tới bằng cây quyết định

# Y_pred_tree = tree.predict([
#     [100,  0.00468,  0.652, 0, 0.941, 0.792,  0.115,  -5.201,  0.0748, 0.759],
#     [27.0,  0.468,  0.652, 0, 0.941, 0.792,  0.115,  -5.201,  0.0748, 0.759],
# ])


# print(Y_pred_tree)
