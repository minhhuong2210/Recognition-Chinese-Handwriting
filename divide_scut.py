import os
import cv2
import json
import shutil
import numpy as np
from PIL import Image

dataset_path = 'C:/Users/hp/Documents/Nam3/NCKH/SCUT-HCCDoc_Dataset_Release_v2' # eg, '/home/dataset/SCUT-HCCDoc_Dataset_Release_v2'
save_path = 'C:/Users/hp/Documents/Nam3/NCKH/data_split' # eg, '/home/dataset/my_path'

def check_save_path():
#kiểm tra xem thư mục lưu ảnh đã xử lý có tồn tại hay không và nếu có, 
#sẽ nhắc người dùng xóa thư mục đó hoặc sửa đổi đường dẫn lưu. 
#Sau đó, nó tạo thư mục và các thư mục con cho các tập huấn luyện, xác thực và kiểm tra.
    if os.path.exists(save_path):
        answer = input(f'Path [{save_path}] exists! Do you want to remove it? [y/n]')
        if answer.strip() == 'y':
            shutil.rmtree(save_path)
        else:
            assert False,'Please modify the save_path!'

    print('Create new directory for saving image: {}'.format(save_path))
    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, 'train_image'))
    os.mkdir(os.path.join(save_path, 'validation_image'))
    os.mkdir(os.path.join(save_path, 'test_image'))

# crop text regions
def image_process(img_path, tl, tr, br, bl):
#lấy một đường dẫn hình ảnh và bốn điểm xác định vị trí của các góc vùng văn bản trong hình ảnh.
#Nó sử dụng OpenCV để biến đổi phối cảnh của vùng văn bản sao cho nó trở thành một hình chữ nhật, 
#sau đó chuyển đổi mảng có nhiều mảng thành một đối tượng hình ảnh PIL.
    img = cv2.imread(img_path)
    width = min(tr[0]-tl[0], br[0]-bl[0])
    height = min(bl[1]-tl[1], br[1]-tr[1])
    point_0 = np.float32([tl, tr, bl, br])
    point_i = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    transform = cv2.getPerspectiveTransform(point_0, point_i)
    img_i = cv2.warpPerspective(img, transform, (width, height))
    return Image.fromarray(img_i)

#Các hàm generate_train_validation_dataset()và generate_test_dataset()tải dữ liệu chú thích từ hai tệp JSON riêng biệt(hccdoc_train.jsonvàhccdoc_test.json) 
#và lặp qua từng cơ sở dữ liệu và phiên bản dữ liệu để trích xuất các vùng văn bản và nhãn(do người dùng đặt) tương ứng. 
#Các hình ảnh và nhãn được trích xuất được lưu trong các thư mục thích hợp và các được ghi vào các gt.txttệp tương ứng.
def generate_train_validation_dataset():

    print('Start to construct the training and validation sets!')
    print('After the construction, the training set should contain 74,603 samples and the validation set should contain 18,551 samples.')
    train_save_path = os.path.join(save_path, 'train_image')
    validation_save_path = os.path.join(save_path, 'validation_image')
    image_path = os.path.join(dataset_path, 'image')

    scut_file = open(os.path.join(dataset_path, 'hccdoc_train.json'),encoding="utf-8")
    results = json.load(scut_file)
    print('Finish loading json file of scut dataset!')

    f_train = open(os.path.join(train_save_path, 'gt.txt'), 'w+',encoding="utf-8")
    f_validtion = open(os.path.join(validation_save_path, 'gt.txt'), 'w+',encoding="utf-8")
    five_keys = results['annotations'].keys()

    cnt = 0
    for key in five_keys:
        database = results['annotations'][key]
        for data in database:
            file_path = os.path.join(image_path, data['file_path'])
            gts = data['gt']
            for index, gt in enumerate(gts):
                point, text = gt['point'], gt['text']
                crop_img = image_process(file_path, point[0:2], point[2:4], point[4:6], point[6:8])

                if index % 5 == 0: # 0,5,10,... dc vao validation
                    crop_img.save(os.path.join(validation_save_path, '{}.jpg'.format(cnt)))
                    f_validtion.write('{} {}\n'.format(os.path.join(train_save_path, '{}.jpg'.format(cnt)), text.replace(' ', '')))
                else: # con lai thi dc train
                    crop_img.save(os.path.join(train_save_path, '{}.jpg'.format(cnt)))
                    f_train.write('{} {}\n'.format(os.path.join(train_save_path, '{}.jpg'.format(cnt)), text.replace(' ', '')))

                cnt += 1
                if cnt % 1000 == 0:
                    print(cnt)

    f_train.close()
    f_validtion.close()


def generate_test_dataset():
    print('Start to construct the testing set!')
    print('After the construction, the testing set should contain 23,389 samples.')
    test_save_path = os.path.join(save_path, 'test_image')
    image_path = os.path.join(dataset_path, 'image')

    scut_file = open(os.path.join(dataset_path, 'hccdoc_test.json'))
    results = json.load(scut_file)
    print('Finish loading json file of scut dataset!')

    f_test = open(os.path.join(test_save_path, 'gt.txt'), 'w+',encoding="utf-8")
    five_keys = results['annotations'].keys()

    cnt = 0
    for key in five_keys:
        database = results['annotations'][key]
        for data in database:
            file_path = os.path.join(image_path, data['file_path'])
            gts = data['gt']
            for index, gt in enumerate(gts):
                point, text = gt['point'], gt['text']
                crop_img = image_process(file_path, point[0:2], point[2:4], point[4:6], point[6:8])

                crop_img.save(os.path.join(test_save_path, '{}.jpg'.format(cnt)))
                f_test.write('{} {}\n'.format(os.path.join(test_save_path, '{}.jpg'.format(cnt)), text.replace(' ', '')))

                cnt += 1
                if cnt % 1000 == 0:
                    print(cnt)

    f_test.close()
#Tập lệnh in một số thông tin ra bảng điều khiển trong quá trình xử lý, 
#chẳng hạn như số lượng mẫu được xử lý và số lượng mẫu dự kiến ​​trong mỗi bộ. 
#Cuối cùng, nó in một thông báo thành công khi quá trình xử lý hoàn tất.
if __name__ == '__main__':
    # check_save_path()
    # generate_train_validation_dataset()
    generate_test_dataset()
    print('Successfully loading!')
