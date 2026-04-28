import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
import torch.nn.functional as F
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import piq
import json
from tqdm import tqdm
import copy

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    max_pixel = 1.0
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_value


def fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid_out = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid_out


def analysis(pre_image, GT_image):
    MSE = float(F.mse_loss(pre_image, GT_image))
    SSIM = float(piq.ssim(pre_image, GT_image, data_range=1.0))
    PSNR = float(psnr(pre_image, GT_image))
    FID = float(fid(pre_image.numpy()[0][0], GT_image.numpy()[0][0]))

    return MSE, SSIM, PSNR, FID


def evaluation_metrics(pre_path, mode='overlap', figure=False, figure_save_path='', json_save_root=''):
    if mode == 'overlap':
        gt_path = ROOT_PATH + '/Result_Data/ground_truth/overlap'
    else:
        gt_path = ROOT_PATH + '/Result_Data/ground_truth/nonoverlap'

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # input GT & predict
    image_path = '{}/recon_image'.format(gt_path)
    image_list = os.listdir(image_path)
    data_valid_list = []
    for index, image_name in tqdm(enumerate(image_list), desc="Image List ({})".format(len(image_list))):
        # print('{} / {}, {}'.format(index + 1, len(image_list), image_name))
        GT_image = np.array(Image.open('{}/recon_image/{}'.format(gt_path, image_name))) / 255
        GT_upper_layer = np.array(Image.open('{}/upper_layer/{}'.format(gt_path, image_name))) / 255
        GT_lower_layer = np.array(Image.open('{}/lower_layer/{}'.format(gt_path, image_name))) / 255
        GT_upper_mask = np.array(Image.open('{}/upper_mask/{}.bmp'.format(gt_path, image_name[:-4]))) / 255
        GT_lower_mask = np.array(Image.open('{}/lower_mask/{}.bmp'.format(gt_path, image_name[:-4]))) / 255

        recon_image = np.array(Image.open('{}/recon_image/{}'.format(pre_path, image_name))) / 255
        upper_layer = np.array(Image.open('{}/upper_layer/{}'.format(pre_path, image_name))) / 255
        lower_layer = np.array(Image.open('{}/lower_layer/{}'.format(pre_path, image_name))) / 255
        upper_mask = np.array(Image.open('{}/upper_mask/{}.bmp'.format(pre_path, image_name[:-4]))) / 255
        lower_mask = np.array(Image.open('{}/lower_mask/{}.bmp'.format(pre_path, image_name[:-4]))) / 255
        soft_tissue = np.array(Image.open('{}/soft_tissue/{}'.format(pre_path, image_name))) / 255

        GT_image = transform(Image.fromarray(GT_image)).unsqueeze(0)
        recon_image = transform(Image.fromarray(recon_image)).unsqueeze(0)
        soft_tissue = transform(Image.fromarray(soft_tissue)).unsqueeze(0)

        GT_upper_layer = transform(Image.fromarray(GT_upper_layer)).unsqueeze(0)
        GT_lower_layer = transform(Image.fromarray(GT_lower_layer)).unsqueeze(0)
        upper_layer = transform(Image.fromarray(upper_layer)).unsqueeze(0)
        lower_layer = transform(Image.fromarray(lower_layer)).unsqueeze(0)

        overlap = GT_upper_mask + GT_lower_mask
        overlap[overlap != 2] = 0
        overlap_count = np.sum(overlap / 2)

        # joint = image_name.split('_')[2]

        if mode == 'overlap':
            row_list = {'image_name': image_name,
                        'joint': 1,
                        'recon_image': list(analysis(recon_image, GT_image)),
                        'Overlap': int(overlap_count)
                        }
        else:
            row_list = {'image_name': image_name,
                        'joint': 1,
                        'recon_image': list(analysis(recon_image, GT_image)),
                        'upper_image': list(analysis(upper_layer, GT_upper_layer)),
                        'lower_image': list(analysis(lower_layer, GT_lower_layer)),
                        'Overlap': int(overlap_count)
                        }

        data_valid_list.append(row_list)

        if figure:
            fig, axes = plt.subplots(1, 6, figsize=(15, 3.5))
            plt.rcParams.update({
                'font.size': 15,  # 字体大小
                'font.family': 'serif',  # 字体类型
                'font.serif': ['Times New Roman'],  # 字体家族中的具体字体
            })
            plt.subplots_adjust(wspace=0.01, hspace=0, left=0, right=1, top=1, bottom=0)

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            axes[0].imshow(GT_image[0][0], cmap='gray')
            axes[0].set_title('Overlap size: {}'.format(int(overlap_count)))
            axes[0].axis('off')

            axes[1].imshow(recon_image[0][0], cmap='gray')
            # axes[1].set_title('Recon Image')
            axes[1].axis('off')

            axes[2].imshow(upper_layer[0][0], cmap='gray')
            # axes[2].set_title('Upper Layer')
            axes[2].axis('off')

            axes[3].imshow(lower_layer[0][0], cmap='gray')
            # axes[3].set_title('Lower Layer')
            axes[3].axis('off')

            axes[4].imshow(soft_tissue[0][0], cmap='gray')
            # axes[3].set_title('Lower Layer')
            axes[4].axis('off')

            axes[5].imshow(F.mse_loss(recon_image, GT_image, reduction='none')[0][0], vmin=0, vmax=1)
            axes[5].set_title(f'MSE (1e-4): {F.mse_loss(recon_image, GT_image) * 1e4:.2f}')
            axes[5].axis('off')

            plt.savefig('{}/{}'.format(figure_save_path, image_name), bbox_inches='tight', dpi=300)
            plt.close()

    parts = figure_save_path.strip('/').split('/')
    json_name = "_".join(parts[-2:])

    with open('{}/{}.json'.format(json_save_root, json_name), 'w', encoding='utf-8') as json_file:
        json.dump(data_valid_list, json_file, ensure_ascii=False, indent=4)

    return data_valid_list


def data_analysis(file_path, mode='overlap'):
    with open(file_path, 'r') as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    # Calculating means for recon_image, upper_image, lower_image
    base_info_df = df[['joint', 'Overlap']].apply(pd.to_numeric, errors='coerce')

    if mode == 'overlap':
        recon_image_df = pd.DataFrame(df['recon_image'].tolist())
        recon_image_df = pd.concat([base_info_df, recon_image_df], axis=1).apply(pd.to_numeric, errors='coerce')
        recon_image_df.columns = ['Joint', 'Overlap', 'MSE', 'SSIM', 'PSNR', 'FID']

        return recon_image_df
    else:
        recon_image_df = pd.DataFrame(df['recon_image'].tolist())
        recon_image_df = pd.concat([base_info_df, recon_image_df], axis=1).apply(pd.to_numeric, errors='coerce')
        recon_image_df.columns = ['Joint', 'Overlap', 'MSE', 'SSIM', 'PSNR', 'FID']

        upper_image_df = pd.DataFrame(df['upper_image'].tolist())
        upper_image_df = pd.concat([base_info_df, upper_image_df], axis=1).apply(pd.to_numeric, errors='coerce')
        upper_image_df.columns = ['Joint', 'Overlap', 'MSE', 'SSIM', 'PSNR', 'FID']

        lower_image_df = pd.DataFrame(df['lower_image'].tolist())
        lower_image_df = pd.concat([base_info_df, lower_image_df], axis=1).apply(pd.to_numeric, errors='coerce')
        lower_image_df.columns = ['Joint', 'Overlap', 'MSE', 'SSIM', 'PSNR', 'FID']

        return recon_image_df, upper_image_df, lower_image_df


##显示结果
def to_latex(df_org):
    df = df_org.groupby('Joint').agg(['mean', 'std'])
    df = df.sort_values(by='Joint', ascending=False)
    df_all = df_org.agg(['mean', 'std'])

    finger_names = ["Thumb", "Index", "Middle", "Ring", "Small"]
    df["Joint"] = finger_names

    latex_data = {
        "Joint": df["Joint"],
        "MSE": (df["MSE"]['mean'] * 1e4).round(4).astype(str) + " $\\pm$ " + (df['MSE']['std'] * 1e4).round(4).astype(
            str),
        "SSIM": df["SSIM"]['mean'].round(4).astype(str) + " $\\pm$ " + df["SSIM"]['std'].round(4).astype(str),
        "PSNR": df["PSNR"]['mean'].round(4).astype(str) + " $\\pm$ " + df["PSNR"]['std'].round(4).astype(str),
        "FID": df["FID"]['mean'].round(4).astype(str) + " $\\pm$ " + df["FID"]['std'].round(4).astype(str),
        "Overlap": df["Overlap"]['mean'].round(4).astype(str) + " $\\pm$ " + df["Overlap"]['std'].round(4).astype(str),
    }
    latex_data_all = {
        "Joint": 'Overall',
        "MSE": (df_all["MSE"]['mean'] * 1e4).round(4).astype(str) + " $\\pm$ " + (df_all['MSE']['std'] * 1e4).round(
            4).astype(str),
        "SSIM": df_all["SSIM"]['mean'].round(4).astype(str) + ' $\\pm$ ' + df_all["SSIM"]['std'].round(4).astype(str),
        "PSNR": df_all["PSNR"]['mean'].round(4).astype(str) + " $\\pm$ " + df_all["PSNR"]['std'].round(4).astype(str),
        "FID": df_all["FID"]['mean'].round(4).astype(str) + " $\\pm$ " + df_all["FID"]['std'].round(4).astype(str),
        "Overlap": df_all["Overlap"]['mean'].round(4).astype(str) + " $\\pm$ " + df_all["Overlap"]['std'].round(
            4).astype(
            str),
    }

    df1 = pd.DataFrame(latex_data)
    df2 = pd.DataFrame([latex_data_all])

    # 合并两个 DataFrame
    merged_df = pd.concat([df1, df2])
    latex_str = '& MSE & SSIM & PSNR & FID & Overlap Size \\\\\n'
    latex_str += "\\midrule\n"
    for index, row in merged_df.iterrows():
        latex_str += " & ".join(map(str, row.values)) + " \\\\\n"

    latex_str += "\\tabular"

    return latex_str


def figure_output(config, save_path=''):
    df_list = config['df_list']
    color_list = config['color_list']
    method_list = config['method_list']
    sci_setting = config['sci_setting']
    def sorted_plot(x, y, window=50):
        sorted_indices = sorted(range(len(x)), key=lambda k: x[k])
        x = [x[i] for i in sorted_indices]
        y = [y[i] for i in sorted_indices]
        non_zero_indices = [i for i, value in enumerate(x) if value != 0]
        zero_indices = [i for i, value in enumerate(x) if value == 0]

        x_zero = np.mean([x[i] for i in zero_indices])
        y_zero = np.mean([y[i] for i in zero_indices])
        y_min_zero = np.quantile([y[i] for i in zero_indices], q=0)
        y_max_zero = np.quantile([y[i] for i in zero_indices], q=1)

        x_non_zero = [x[i] for i in non_zero_indices]
        y_non_zero = [y[i] for i in non_zero_indices]
        x_list, y_list, y_min_list, y_max_list = [x_zero], [y_zero], [y_min_zero], [y_max_zero]

        for i in range(len(x_non_zero) - window):
            x_temp_list = x_non_zero[i: i + window]
            y_temp_list = sorted(y_non_zero[i: i + window])
            x_list.append(np.mean(x_temp_list))
            y_list.append(np.mean(y_temp_list))
            y_min_list.append(np.mean(y_temp_list[:len(y_temp_list) // 2]))
            y_max_list.append(np.mean(y_temp_list[len(y_temp_list) // 2:]))
        return x_list, y_list, y_min_list, y_max_list

    font_family = 'Times New Roman'
    label_font = {'family': font_family, 'weight': 'normal', 'size': 15}
    metrics = ['MSE', 'PSNR', 'SSIM', "FID"]

    plt.figure(figsize=(12, 3.5))
    plt.rcParams['lines.linewidth'] = 2
    plt.subplot(1, len(metrics) + 1, 1)
    x = df_list[0]['Overlap']
    x_log = copy.deepcopy(x)
    x_log[x_log == 0] = 0.6703200460356393
    x_log = np.log10(x_log)
    range_list = [-0.4] + [i * 0.4 for i in range(11)]
    plt.hist(x_log, range_list, color='skyblue', edgecolor='black', linewidth=0.5, align='mid')
    for i, patch in enumerate(plt.gca().patches):
        patch.set_facecolor(color_list[min(i, 1)])
    plt.xlabel('Overlap size(log(x))', fontdict=label_font)
    plt.yscale('log')
    plt.title('(A) Distribution', fontdict=label_font)
    plt.grid(True)

    metrics_index = 0

    for idx, metric in enumerate(metrics, start=2):
        plt.subplot(1, len(metrics) + 1, idx)
        df_index = 0
        for df, color in zip(df_list, color_list):
            x_list, y_list, _, _ = sorted_plot(x, df[metric])
            y_list = [x * (10 ** sci_setting[metrics_index]) for x in y_list]
            plt.plot(x_list, y_list, color=color, label=f'{method_list[df_index]}')
            df_index += 1
        ax = plt.gca()
        if sci_setting[metrics_index] != 0:
            plt.text(0, 1, '1e-{}'.format(sci_setting[metrics_index]), transform=ax.transAxes,
                     fontsize=10, ha='left', va='bottom',
                     bbox=dict(facecolor='white', alpha=0, edgecolor='none'))
        plt.xlabel('Overlap size', fontdict=label_font)
        plt.title(f'({chr(64 + idx)}) {metric.upper()}', fontdict=label_font)
        plt.grid(True)
        metrics_index += 1
    plt.legend(prop={'family': font_family, 'size': 14}, bbox_to_anchor=(1.05, 1.35), ncol=len(method_list))
    plt.subplots_adjust(left=0.03, right=0.99, wspace=0.30, bottom=0.15, top=0.75)
    # plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # # overlap
    # mode = 'overlap'
    # pre_path = ROOT_PATH + '/Result_Data/pre_result/overlap'
    # save_path = ROOT_PATH + '/Result_Figure/pre_overlap'
    # evaluation_list = evaluation_metrics(pre_path, mode=mode, figure=True, save_path=save_path)
    #
    # # nonoverlap
    mode = 'nonoverlap'
    # pre_path = ROOT_PATH + '/Result_Data/pre_result/nonoverlap'
    # save_path = ROOT_PATH + '/Result_Figure/pre_nonoverlap'
    # evaluation_list = evaluation_metrics(pre_path, mode=mode, figure=True, save_path=save_path)

    test_path = '/Users/wanghaolin/PycharmProjects/BLS_GAN_v2/experiments/Exp_generation/results/results_pre_nonoverlap.json'

    recon_image_df, upper_image_df, lower_image_df = data_analysis(test_path, mode=mode)

    config = {'df_list': [recon_image_df, upper_image_df, lower_image_df, lower_image_df * 2],
              'color_list': ['limegreen', 'slateblue', 'deepskyblue', 'deepskyblue'],
              'method_list': ['A', 'B', 'C', 'C'],
              'sci_setting': [6, 0, 1, 3]
              }
    figure_output(config)
