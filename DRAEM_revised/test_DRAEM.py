import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc," + run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc," + run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap," + run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap," + run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt", 'a+') as file:
        file.write(fin_str)


def visualize_samples(run_name, samples):
    output_dir = os.path.join("./outputs", run_name)
    os.makedirs(output_dir, exist_ok=True)

    for idx, (original_image, reconstructed_image, ground_truth, predicted_mask) in enumerate(samples):
        if idx >= 7:  # Visualize only the first 7 samples
            break

        # Convert tensors to NumPy for visualization
        original_image = original_image.cpu().numpy().transpose(1, 2, 0)
        reconstructed_image = reconstructed_image.cpu().numpy().transpose(1, 2, 0)
        ground_truth = ground_truth.cpu().numpy().squeeze()
        predicted_mask = (predicted_mask.cpu().numpy().squeeze() > 0.5).astype(int)

        # Plot and save the visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(np.clip(original_image, 0, 1))
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(np.clip(reconstructed_image, 0, 1))
        axes[1].set_title("Reconstructed Image")
        axes[1].axis("off")

        axes[2].imshow(ground_truth, cmap="gray")
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis("off")

        axes[3].imshow(predicted_mask, cmap="jet")
        axes[3].set_title("Predicted Mask (Binary)")
        axes[3].axis("off")

        plt.savefig(os.path.join(output_dir, f"sample_{idx + 1}.png"))
        plt.close()
        print(f"Saved visualization for sample {idx + 1} to {output_dir}")


def test(obj_names, mvtec_path, checkpoint_path, base_model_name):
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    visualization_samples = []  # Collect samples for visualization

    for obj_name in obj_names:
        img_dim = 256
        run_name = base_model_name + "_" + obj_name + '_'

        model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name + ".pckl"), map_location='cuda:0'))
        model.cuda()
        model.eval()

        model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name + "_seg.pckl"), map_location='cuda:0'))
        model_seg.cuda()
        model_seg.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        total_pixel_scores = []
        total_gt_pixel_scores = []
        anomaly_score_gt = []
        anomaly_score_prediction = []

        with torch.no_grad():
            for i_batch, sample_batched in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Testing {obj_name}"):
                gray_batch = sample_batched["image"].cuda()
                is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
                anomaly_score_gt.append(is_normal)
                true_mask = sample_batched["mask"].cpu().numpy().squeeze()

                gray_rec = model(gray_batch)
                joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

                out_mask = model_seg(joined_in)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                if torch.isnan(out_mask_sm).any():
                    print(f"NaN detected in output mask for batch {i_batch}. Skipping...")
                    continue

                out_mask_np = out_mask_sm[:, 1, :, :].detach().cpu().numpy().squeeze()
                total_pixel_scores.extend(out_mask_np.flatten())
                total_gt_pixel_scores.extend(true_mask.flatten())

                image_score = np.max(out_mask_np)
                anomaly_score_prediction.append(image_score)

                # Collect samples for visualization
                if len(visualization_samples) < 7:  # Collect up to 7 samples
                    visualization_samples.append((
                        gray_batch[0],
                        gray_rec[0],
                        sample_batched["mask"][0],
                        out_mask_sm[0, 1, :, :]
                    ))

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        total_pixel_scores = np.array(total_pixel_scores)
        total_gt_pixel_scores = np.array(total_gt_pixel_scores)

        total_gt_pixel_scores = (total_gt_pixel_scores > 0.5).astype(int)

        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)

        print(f"{obj_name}\nAUC Image: {auroc:.4f}\nAP Image: {ap:.4f}\nAUC Pixel: {auroc_pixel:.4f}\nAP Pixel: {ap_pixel:.4f}\n==============================")

    print(run_name)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))

    write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)

    
    visualize_samples(run_name, visualization_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)

    args = parser.parse_args()

    # obj_list = ['bottle']
    obj_list = ['leather']


    with torch.cuda.device(args.gpu_id):
        test(obj_list, args.data_path, args.checkpoint_path, args.base_model_name)


# python test_DRAEM.py --gpu_id 1 --base_model_name DRAEM_test_0.0001_300_bs1_bottle_ --data_path ../datasets/mvtec/ --checkpoint_path ./checkpoints/bottle_perlin_checkpoints_draem