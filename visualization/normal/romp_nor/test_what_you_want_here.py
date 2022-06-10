def test_pw3d():
    import torch
    from lib.dataset.mixed_dataset import SingleDataset,MixedDataset
    from torch.utils.data import Dataset, DataLoader
    from config import args
    import cv2

    args().configs_yml = 'configs/v1.yml'
    args().dataset = 'h36m'

    # dataset = SingleDataset("pw3d")
    dataset = MixedDataset(train_flag=True)
    dataloader = DataLoader(dataset = dataset,batch_size = 1, shuffle = True)

    count = 0
    target_count = 1

    for data in dataset:

        # 打印
        if count == 0:
            for key in data.keys():
                if type(data[key]) == torch.Tensor:
                    print(key,'\t',data[key].shape)
                else:print(key,'\t',data[key])

        # 图片
        # print(data['imgpath'])
        # for i in range(3):
        #     cv2.imshow(str(i+1),data['image'][i].data.numpy()/255)
        #     cv2.waitKey(0)
        # print()

        count += 1
        if count >= target_count:exit()

def test_h36m():
    import numpy as np

    annot_file_path = 'D:/3D/lihaoyuan/vomp/dataset/h36m/annots.npz'
    annot_file = np.load(annot_file_path,allow_pickle=True)['annots'][()]

    info = annot_file[list(annot_file.keys())[0]].copy()

    # kp3d_mono(32, 3)
    # kp2d(32, 2)
    # kp3d(32, 3)
    # cam(3, 3)
    # poses(3, 72)
    # betas(10, )
    # trans(3, 3)
    for key in info.keys():
        print(key,info[key].shape)


    # print(info['poses'])
    # print(info['betas'])

def test_render(pose,betas,trans,org_img):
    from lib.visualization.visualization import Visualizer
    import torch
    from lib.models.smpl import SMPL
    from lib.config import args

    pose = torch.tensor(pose).cuda()
    betas = torch.tensor(betas).cuda()
    trans = torch.tensor(trans).cuda()

    smpl = SMPL(args().smpl_model_path, J_reg_extra9_path=args().smpl_J_reg_extra_path, J_reg_h36m17_path=args().smpl_J_reg_h37m_path, \
                batch_size=args().batch_size,model_type='smpl', gender='neutral', use_face_contour=False, ext='npz',flat_hand_mean=True, use_pca=False,\
                ).cuda()
    visualizer = Visualizer(resolution=(512,512), result_img_dir = 'D:/3D/lihaoyuan/vomp/dataset/trash/', renderer_type='pyrender')

    def test_smpl():
        T_params_dict = {}
        T_params_dict['poses'] = torch.zeros(1,72)
        T_params_dict['betas'] = torch.zeros(1,10)
        T_smpl_outs = smpl(**T_params_dict, return_verts=True) # ['verts', 'j3d', 'joints_smpl24', 'joints_h36m17']
        print(T_smpl_outs['verts'].shape)

    T_params_dict = {}
    T_params_dict['poses'] = pose
    T_params_dict['betas'] = betas
    T_smpl_outs = smpl(**T_params_dict, return_verts=True)

    per_img_verts_list = T_smpl_outs['verts'].unsqueeze(0)  # [1,1,6890,3]
    mesh_trans = trans.unsqueeze(0)  # [1,1,3]
    # org_imgs [1,512,512,3]
    rendered_imgs = visualizer.visualize_renderer_verts_list(per_img_verts_list, images=org_img.copy(),
                                                             trans=mesh_trans)

    return  rendered_imgs

def gen_h36m_2d_image():
    import cv2
    import numpy as np
    import os

    print('begin to open annot file...')
    info_file_path = 'C:/Users/Public/Desktop/info.npz'
    info_exist = os.path.exists(info_file_path) # and False

    if not info_exist:
        annot_file_path = 'D:/3D/lihaoyuan/vomp/dataset/h36m/annots.npz'
        annot_file = np.load(annot_file_path, allow_pickle=True)['annots'][()]

    root = 'D:/3D/lihaoyuan/vomp/dataset/h36m/images/S1_Directions 1_0_'
    suffix = '.jpg'
    for i in range(274 + 1):
        img_path = root + str(i) + suffix
        base_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)

        if info_exist:
            info = np.load(info_file_path, allow_pickle=True)['info'][()]
        else:
            info = annot_file[base_name].copy()
        # print('successfully get info...')

        # 保存字典
        if i == 0 and not info_exist:
            np.savez(info_file_path, info=info)

        for point in info['kp2d']:
            cv2.circle(img,point,2,(0,255,0),4)


        # cv2.imshow(str(i),img)
        # cv2.waitKey(0)

        cv2.imwrite('D:/3D/lihaoyuan/vomp/dataset/trash/' + str(i) + '.jpg', img)

        # exit()

def gen_h36m_3d_image():
    import cv2
    import numpy as np
    import os

    print('begin to open annot file...')
    info_file_path = 'C:/Users/Public/Desktop/info.npz'
    info_exist = os.path.exists(info_file_path) and False

    if not info_exist:
        annot_file_path = 'D:/3D/lihaoyuan/vomp/dataset/h36m/annots.npz'
        annot_file = np.load(annot_file_path, allow_pickle=True)['annots'][()]

    root = 'D:/3D/lihaoyuan/vomp/dataset/h36m/images/S1_Directions 1_0_'
    suffix = '.jpg'
    for i in range(274+1):
        img_path = root + str(i) + suffix
        base_name = img_path.split('/')[-1]
        img = np.reshape(cv2.resize(cv2.imread(img_path),(512,512)),(1,512,512,3))

        if info_exist:
            info =  np.load(info_file_path,allow_pickle=True)['info'][()]
        else:
            info = annot_file[base_name].copy()
        # print('successfully get info...')

        # 保存字典
        if i == 0 and not info_exist:
            np.savez(info_file_path,info=info)

        pose = info['poses'][:1]
        pose[:,:3] = info['cam'][:1]

        info['trans'][0][2] += 3    # 大小
        info['trans'][0][1] -= 0.6  # 上
        info['trans'][0][0] -= 0.03  # 左
        # print(info['trans'])

        render_img = test_render(pose=pose,betas=np.reshape(info['betas'],(1,10)),trans=info['trans'][:1],\
                                 org_img=img)

        render_img = np.array(render_img)
        # print(render_img.shape)
        # cv2.imshow(str(i),render_img[0])
        # cv2.waitKey(0)

        cv2.imwrite('D:/3D/lihaoyuan/vomp/dataset/trash/'+str(i)+'.jpg',render_img[0])

        # exit()

def get_video_from_img():
    import os
    import cv2
    from tqdm import tqdm

    image_path = 'D:/3D/lihaoyuan/vomp/dataset/trash/img/'
    store_path = 'C:/Users/Administrator/Desktop/'
    video_name = 'output'
    fps = 15

    picture_in_filelist = os.listdir(image_path)
    suffix = picture_in_filelist[1].split('.')[1]
    img_name = image_path + picture_in_filelist[1]
    img = cv2.imread(img_name)
    h,w,c = img.shape
    size = (w,h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = store_path + video_name + '.mp4'
    video_writer = cv2.VideoWriter(out_video,fourcc,fps,size)

    name_list = os.listdir(image_path)
    name_list.sort(key=lambda x:int(x.split('.')[0]))

    for i in tqdm(name_list):
        picture_in_filename = image_path + i
        img_insert = cv2.imread(picture_in_filename)
        video_writer.write(img_insert)
    video_writer.release()

    print("ok!")

def load_j3d():
    import numpy as np

    j3d_path = r'D:\3D\lihaoyuan\predict\ROMP\demo\j3d\sample_video_result.npz'

    annot = np.load(j3d_path,allow_pickle=True)['j3d'][()]
    print(annot.keys())
    print(type(annot['1']))

load_j3d()