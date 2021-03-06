import copy
import random
import json
import os
import numpy as np
import cv2
import tqdm
from PIL import Image


def numpyPSNR(tar_img_dir, prd_img_dir):

    tar_img = cv2.imread(tar_img_dir)
    prd_img = cv2.imread(prd_img_dir)

    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps


def median_imgs(img_dirs):

    imgs_r = []
    imgs_g = []
    imgs_b = []

    for img_dir in img_dirs:
        img = cv2.imread(img_dir)
        imgs_b.append(img[:, :, 0])
        imgs_g.append(img[:, :, 1])
        imgs_r.append(img[:, :, 2])

    imgs_r = np.median(np.stack(imgs_r, axis=0), axis=0)
    imgs_g = np.median(np.stack(imgs_g, axis=0), axis=0)
    imgs_b = np.median(np.stack(imgs_b, axis=0), axis=0)

    imgs = np.stack([imgs_b, imgs_g, imgs_r], axis=2)
    return imgs


class MedianImgs():
    def __init__(self, img_dirs):
        self.imgs_list = []

        imgs_r = []
        imgs_g = []
        imgs_b = []

        for img_dir in img_dirs:
            img = cv2.imread(img_dir)
            self.imgs_list.append(img)
            imgs_b.append(img[:, :, 0])
            imgs_g.append(img[:, :, 1])
            imgs_r.append(img[:, :, 2])

        imgs_r = np.median(np.stack(imgs_r, axis=0), axis=0)
        imgs_g = np.median(np.stack(imgs_g, axis=0), axis=0)
        imgs_b = np.median(np.stack(imgs_b, axis=0), axis=0)

        self.median_img = np.stack([imgs_b, imgs_g, imgs_r], axis=2)

    def get_median_result(self):
        return self.median_img

    def get_imgs_list(self):
        return self.imgs_list


def get_human_forrest_db(DB_dir, target_noise_type, show_details=False, check_json=False):
    ############################################################################################################
    # Figure out Each version, read meta infos (error txt), init
    ############################################################################################################
    print('===========================================================================================')
    print('Total information')
    print('===========================================================================================')
    R_F_D_S_C = [os.path.join(DB_dir, x) for x in sorted(os.listdir(DB_dir))]

    # only directories (except wrong folders)
    R_F_D_S_C = [tempdir for tempdir in R_F_D_S_C if os.path.isdir(tempdir)]

    # todo
    # R_F_D_S_C = R_F_D_S_C[:3]

    # load error json list npy
    error_json_list_dir = 'error_json_list.txt'
    error_json_list = []
    if os.path.isfile(error_json_list_dir):
        with open(error_json_list_dir, "r") as fp:
            error_json_list = json.load(fp)
            print(f'>> Load {error_json_list_dir}')
    else:
        print(f'>> No {error_json_list_dir}')


    # init my_dict_per_version
    my_dict_per_version = []

    # init images_len_sum
    images_len_sum = 0

    # If the DB's json is checked than skip reading this.
    R_F_D_S_C_checked_txt = "R_F_D_S_C_checked.txt"
    R_F_D_S_C_list = []
    if check_json and os.path.isfile(R_F_D_S_C_checked_txt):
        print(f'\n>> Checked DB {error_json_list_dir}')
        f = open(R_F_D_S_C_checked_txt, 'r')
        R_F_D_S_C_checked_lines = f.readlines()
        for line in R_F_D_S_C_checked_lines:
            line = line.strip()  # ??? ?????? ??? ?????? ????????? ????????????.
            print(line)
            R_F_D_S_C_list.append(line)
        f.close()

    ############################################################################################################
    # Read all noises one by one, Check json Errors.
    ############################################################################################################
    for version_dir in R_F_D_S_C:
        if check_json and version_dir in R_F_D_S_C_list:
            lets_check_db = False
        else:
            lets_check_db = True

        if lets_check_db:
            version_base_name = os.path.basename(version_dir)
            DB_list = [os.path.join(version_dir, x) for x in sorted(os.listdir(version_dir))]

            print('\n===========================================================================================')
            print(version_base_name, '---------------------------')
            print('>> Init images len :', len(DB_list))
            images_len_sum += len(DB_list)

            noises = ['R', 'F', 'D', 'S', 'L']  # Rain, Fog, Dust, Snow, Lowlight

            my_dict = {}

            for db_dir in tqdm.tqdm(DB_list[25616:]):

                if check_json and db_dir not in error_json_list:
                    my_json = os.path.splitext(db_dir)[0] + '.json'
                    drawImg = get_sky(my_json)
                    if drawImg is None:
                        error_json_list.append(db_dir)
                        print('json error :', db_dir)

                if db_dir not in error_json_list:
                    # read only image not json.
                    my_format = os.path.splitext(db_dir)[-1]

                    if my_format in ['.jpg', '.png']:

                        # check image size.
                        img_temp = Image.open(db_dir)
                        w, h = img_temp.size
                        size_error = False
                        if w != 1920 or h != 1080:
                            size_error = True
                            print('size error :', db_dir, w, h)


                        if not size_error:
                            # (1) date
                            date = os.path.basename(db_dir).split('_')[0].split('-')[1]
                            # (2) video_num
                            video_num = os.path.basename(db_dir).split('_')[2]
                            # (3)Noise_id
                            noise_id = None

                            info = os.path.basename(db_dir).split('_')[1]
                            for n in noises:
                                if n in info:
                                    noise_id = n
                                    break

                            if noise_id is not None:
                                # (4) place_id
                                place_id = info.split(noise_id)[0]
                                # (5) noise_level
                                noise_level = info.split(noise_id)[1]

                                # This information is combined to create a key.
                                my_key = f"{date}_{place_id}_{video_num}_{noise_id}"

                                if my_key not in my_dict:
                                    my_dict[my_key] = {}
                                    my_dict[my_key][noise_id] = {'01': [], '02': [], '03': [], '04': [], 'GT': []}
                                    my_dict[my_key][noise_id][noise_level].append(db_dir)
                                else:
                                    my_dict[my_key][noise_id][noise_level].append(db_dir)

            # print info
            dict_counter = get_count_for_each_noise(my_dict)
            for my_dict_key2 in dict_counter.keys():
                print(f'>> {my_dict_key2} : {len(dict_counter[my_dict_key2])}')

            my_dict_per_version.append(my_dict)

        if lets_check_db:
            with open(R_F_D_S_C_checked_txt, "a") as f:
                f.write(f"{version_dir}\n")


    ############################################################################################################
    # Merge each version's dict, Show total info.
    ############################################################################################################
    total_dict = {}
    for my_dict_ in my_dict_per_version:
        total_dict.update(my_dict_)

    total_dict_old = copy.deepcopy(total_dict)


    ############################################################################################################
    # Json errors.
    ############################################################################################################
    # save error json list
    print('\n===========================================================================================')
    with open(error_json_list_dir, "w") as fp:
        json.dump(error_json_list, fp)
    print(f'Each json Error information : {len(error_json_list)} (It is excluded from the training data set)')
    for error_json_dir in error_json_list:
        print(error_json_dir)


    ############################################################################################################
    # Dataset errors.
    ############################################################################################################
    # Find error dataset
    total_dict_error = {}
    for my_key in total_dict.keys():
        error = False
        for my_key2 in total_dict[my_key].keys():
            if total_dict[my_key][my_key2]:
                for my_key3 in total_dict[my_key][my_key2].keys():
                    if len(total_dict[my_key][my_key2][my_key3]) == 0:
                        error = True
                    if my_key3 != 'GT' and len(total_dict[my_key][my_key2][my_key3]) < 30:
                        error = True
        if error:
            total_dict_error[my_key] = total_dict[my_key]

    # delete error keys from total_dict
    for my_key in total_dict_error.keys():
        total_dict.pop(my_key)

    # print error dataset
    print('\n===========================================================================================')
    print('Each image Error information (It is excluded from the training data set)')
    for my_key in total_dict_error.keys():
        print('---------------------------')
        print('p_id :', my_key)
        for my_key2 in total_dict_error[my_key].keys():
            if total_dict_error[my_key][my_key2]:
                print('   ', my_key2)
                for my_key3 in total_dict_error[my_key][my_key2].keys():
                    print('      ', my_key3, ':', len(total_dict_error[my_key][my_key2][my_key3]))
                    if my_key3 == 'GT':
                        print('       GT dir :', total_dict_error[my_key][my_key2][my_key3])



    print('\n===========================================================================================')
    print('total size')
    print('>> images len :', images_len_sum)
    dict_counter = get_count_for_each_noise(total_dict_old)
    for my_dict_key2 in dict_counter.keys():
        print(f'>> {my_dict_key2} : {len(dict_counter[my_dict_key2])}')

    print('\ntotal size (After error removal)')
    dict_counter = get_count_for_each_noise(total_dict)
    for my_dict_key2 in dict_counter.keys():
        print(f'>> {my_dict_key2} : {len(dict_counter[my_dict_key2])}')


    ############################################################################################################
    # Option. Show detail infos.
    ############################################################################################################
    if show_details:
        print('\n===========================================================================================')
        print('Each image information')
        for my_key in total_dict.keys():
            print('---------------------------')
            print('p_id :', my_key)
            for my_key2 in total_dict[my_key].keys():
                if total_dict[my_key][my_key2]:
                    print('   ', my_key2)
                    for my_key3 in total_dict[my_key][my_key2].keys():
                        print('      ', my_key3, ':', len(total_dict[my_key][my_key2][my_key3]))
                        if my_key3 == 'GT':
                            print('       GT dir :', total_dict[my_key][my_key2][my_key3])


    ############################################################################################################
    # Return the target Dataset.
    ###########################################################################################################
    my_DB = []
    for my_key in total_dict.keys():
        for my_key2 in total_dict[my_key].keys():
            if total_dict[my_key][my_key2]:
                if my_key2 == target_noise_type:
                    my_DB.append(total_dict[my_key][my_key2])
    return my_DB


def get_count_for_each_noise(my_dict):
    dict_counter = {}
    for my_dict_key in my_dict.keys():
        for my_dict_key2 in my_dict[my_dict_key].keys():
            if my_dict_key2 not in dict_counter:
                dict_counter[my_dict_key2] = [my_dict_key]
            else:
                dict_counter[my_dict_key2].append(my_dict_key)

    return dict_counter


class HumanForrestManager:
    def __init__(self, DB_dir, target_noise_type, show_details=False, check_json=False):
        # ['R', 'F', 'D', 'S', 'L']
        # Example.
        # get L DB
        self.my_db = get_human_forrest_db(DB_dir, target_noise_type, show_details, check_json)

    def get_db_len(self):
        return len(self.my_db)

    def get_input_target_pairs(self, my_idx, noise_level=None, noisy_num=None, median=False):
        if noise_level is None:
            levels = list(self.my_db[my_idx].keys())
            levels.remove('GT')
            level = random.choice(levels)
        else:
            if type(noise_level) == int:
                noise_level = f'0{str(noise_level)}'
            level = noise_level

        if not self.my_db[my_idx][level]:
            print('------------------------------')
            print('Error idx :', my_idx)
            print(self.my_db[my_idx][level])
            print(self.my_db[my_idx])
            print('==============================')
            quit()
        else:
            if median:
                margin = 3
                if noisy_num == None:
                    noisy_num = random.randint(margin, len(self.my_db[my_idx][level])-margin-1)
                    input = []
                    for i in range(margin * 2 + 1):
                        input.append(self.my_db[my_idx][level][noisy_num - margin + i])
                else:
                    input = []
                    for i in range(margin * 2 + 1):
                        input.append(self.my_db[my_idx][level][noisy_num - margin + i])
            else:
                if noisy_num == None:
                    input = random.choice(self.my_db[my_idx][level])
                else:
                    input = self.my_db[my_idx][level][noisy_num]

        target = self.my_db[my_idx]['GT'][0]

        return input, target


def get_sky(json_dir):
    with open(json_dir, 'r') as f:
        json_data = json.load(f)

    # print(json_data.keys())
    # print(json_data['Raw_Data_Info.'])
    # print(json_data['Source_Data_Info.'])
    # print(json_data['Learning_Data_Info.'])

    h, w = json_data['Raw_Data_Info.']['Resolution'].split(',')
    w, h = int(w), int(h)
    # print('w h :', w, h)

    annotation = json_data['Learning_Data_Info.']['Annotation']
    # print(len(annotation))
    # print(annotation)

    img = np.full((w, h, 3), 0, dtype=np.uint8)

    drawImg = None

    for i, anno in enumerate(json_data['Learning_Data_Info.']["Annotation"]):

        # class id max is '38'
        json_class_id = anno['Class_ID'][-2:]
        rgb_val = int(json_class_id) * 5

        json_polygon = anno['segmentation']

        if len(json_polygon) % 2 == 0:

            n = 2
            polygon_split_to_2 = [json_polygon[i * n:(i + 1) * n] for i in
                                  range((len(json_polygon) + n - 1) // n)]

            pts = np.array(polygon_split_to_2, dtype=np.int32)

            color = rgb_val

            #for cnt in json_polygon:
            drawImg = cv2.fillPoly(img, [pts], (color, color, color))

            # The labels are build up layer by layer.
            # cv2.imwrite(f'my_json_{str(i).zfill(3)}_{json_class_id}.png', drawImg)
        else:
            return None

    # sky's rgb value is '10'
    # labels to sky and the others.
    if drawImg is not None:
        drawImg[drawImg != 10] = 0.0
        drawImg[drawImg == 10] = 1.0

    return drawImg


if __name__ == "__main__":
    hf_DB = HumanForrestManager('/hdd1/works/datasets/ssd2/human_and_forest/R_F_D_S_C', 'F',
                                show_details=True, check_json=True)

    print('\n===========================================================================================')
    print('DB len :', hf_DB.get_db_len())
    # idx = 2
    # if hf_DB.get_db_len() <= idx:
    #     print('No Datasets')
    # else:
    #     median = True
    #     if median:
    #         input, target = hf_DB.get_input_target_pairs(idx, noise_level=4, median=median)
    #         print('input, target', len(input), target)
    #
    #         median_imgs = MedianImgs(input)
    #         my_median_img = median_imgs.get_median_result()
    #         my_noisy_img = median_imgs.get_imgs_list()[3]
    #
    #         my_json = os.path.splitext(input[0])[0] + '.json'
    #         drawImg = get_sky(my_json)
    #
    #         sd = 8
    #         truncate = 2
    #         radius = int(truncate * sd + 0.5)
    #         drawImg = cv2.GaussianBlur(drawImg * 255, (radius*2+1, radius*2+1), sd)
    #         drawImg = np.clip(drawImg, 0, 255)
    #         cv2.imwrite('mask.png', drawImg)
    #         drawImg = drawImg / 255
    #
    #         my_gt_img = cv2.imread(target)
    #
    #         clouds = my_median_img * drawImg
    #         forground = my_gt_img * (1 - drawImg)
    #
    #         new_gt_img = clouds + forground
    #
    #
    #         cv2.imwrite('my_noisy_img.png', my_noisy_img)
    #         cv2.imwrite('my_gt_img.png', my_gt_img)
    #         cv2.imwrite('new_gt_img.png', new_gt_img)
    #         cv2.imwrite('median_img.png', my_median_img)
    #
    #         print(numpyPSNR('my_noisy_img.png', 'my_gt_img.png'))
    #         print(numpyPSNR('my_noisy_img.png', 'new_gt_img.png'))
    #     else:
    #         input, target = hf_DB.get_input_target_pairs(idx, noise_level=4, median=median)
    #         print('input, target', len(input), target)
    #
    #         my_json = os.path.splitext(input)[0] + '.json'
    #         drawImg = get_sky(my_json)
    #         cv2.imwrite('mask.png', (drawImg * 255))
    #
    #         my_noisy_img = cv2.imread(input)
    #         my_gt_img = cv2.imread(target)
    #
    #         clouds = my_noisy_img * drawImg
    #         forground = my_gt_img * (1 - drawImg)
    #         new_gt_img = clouds + forground
    #
    #         cv2.imwrite('my_noisy_img.png', my_noisy_img)
    #         cv2.imwrite('my_gt_img.png', my_gt_img)
    #         cv2.imwrite('new_gt_img.png', new_gt_img)
    #
    #         print(numpyPSNR('my_noisy_img.png', 'my_gt_img.png'))
    #         print(numpyPSNR('my_noisy_img.png', 'new_gt_img.png'))






