# bright_blue

全てpython3用

# 元動画の用意
MacならiPhone・iPadを接続してQuickTimePlayerで、ファイル→新規ムービー収録→録画ボタン横の「v」から接続してデバイスを選んで録画。
Windowsは各自で準備。

# 動画から顔画像の切り出し
検出器をダウンロードしてvideo2face.pyと同じディレクトリに配置
http://anime.udp.jp/data/lbpcascade_animeface.xml

引数に変換したい動画ファイルを指定してvideo2face.pyを実行

# 実行用データ
facesフォルダ下に分類別のフォルダを作り、その中にtestとtrainフォルダを作ってテスト用画像・訓練用画像を入れる。
label_dictをfacesフォルダ下と合わせる。

サンプルデータ
https://drive.google.com/file/d/0B5BZ3u1csFVTZTRRUHd5eWdjY2c/view?usp=sharing


# 訓練
train.pyを実行

# 顔判定
predict.pyと同じディレクトリにexperimentフォルダを作成しその中に画像を入れてpredict.pyを実行
label_dictのインデックスに対応している


## 実行例

label_dictのインデックスに対応している
label_dict = {
    'fumika':0, 'hajime':1, 'miyu':2, 'nono':3,'other':4, 'yumi':5
}

0.2026938796
0.7885760069 <- 約80%でhajime
0.0084930342
0.0001763824
0.0000135761
0.0000471687