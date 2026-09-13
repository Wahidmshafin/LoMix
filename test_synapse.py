import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

from lib.networks import EMCADNet
from utils.dataset_synapse import Synapse_dataset


def calculate_dice(pred, gt):
    """
    Binary Dice calculation matching official Synapse / TransUNet benchmark logic
    (utils.utils.calculate_metric_percase / calculate_dice_percase).
    """
    pred = (pred > 0)
    gt = (gt > 0)
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum > 0 and gt_sum > 0:
        intersection = np.logical_and(pred, gt).sum()
        return (2.0 * intersection) / float(pred_sum + gt_sum)
    elif pred_sum > 0 and gt_sum == 0:
        return 1.0
    else:
        # Matches official logic: both empty or missed structure yields 0.0
        return 0.0


def calculate_hd95(pred, gt):
    """
    HD95 distance matching official Synapse / TransUNet benchmark logic
    (utils.utils.calculate_metric_percase).
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum > 0 and gt_sum > 0:
        try:
            from medpy import metric
            return metric.binary.hd95(pred, gt)
        except Exception:
            return 0.0
    elif pred_sum > 0 and gt_sum == 0:
        return 0.0
    else:
        # Matches official logic: both empty or missed structure yields 0.0
        return 0.0


def evaluate_volume(image, label, net, classes, patch_size=(224, 224), eval_hd95=False):
    """
    Slice-by-slice 3D volume inference.
    """
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    prediction = np.zeros_like(label)

    if len(image.shape) == 3:
        for ind in range(image.shape[0]):
            slice_img = image[ind, :, :]
            x, y = slice_img.shape[0], slice_img.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice_img_resized = zoom(slice_img, (patch_size[0] / x, patch_size[1] / y), order=3)
            else:
                slice_img_resized = slice_img

            inp = torch.from_numpy(slice_img_resized).unsqueeze(0).unsqueeze(0).float().cuda()
            with torch.no_grad():
                P = net(inp)
                outputs = P[-1] if isinstance(P, list) else P
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()

                if x != patch_size[0] or y != patch_size[1]:
                    pred_slice = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred_slice = out
                prediction[ind] = pred_slice
    else:
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            P = net(inp)
            outputs = P[-1] if isinstance(P, list) else P
            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().detach().numpy()

    case_dices = []
    case_hd95s = []
    for c in range(1, classes):
        d = calculate_dice(prediction == c, label == c)
        case_dices.append(d)
        if eval_hd95:
            h = calculate_hd95(prediction == c, label == c)
            case_hd95s.append(h)
        else:
            case_hd95s.append(0.0)

    return case_dices, case_hd95s, prediction


def main():
    parser = argparse.ArgumentParser(description="Fast Synapse benchmark evaluation")
    parser.add_argument('--volume_path', type=str, default='./data/synapse/test_vol_h5_new',
                        help='Root directory for test volume .npy.h5 data')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse',
                        help='Directory containing test_vol.txt')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the trained checkpoint (e.g., fixed_best.pth or best.pth)')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of classes including background (default 9)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image resolution (default 224)')
    parser.add_argument('--encoder', type=str, default='pvt_v2_b2',
                        help='Encoder architecture: pvt_v2_b2, resnet34, etc.')
    parser.add_argument('--activation_mscb', type=str, default='relu6',
                        help='Activation used in MSCB (default: relu6)')
    parser.add_argument('--eval_hd95', action='store_true', default=False,
                        help='Enable 3D HD95 calculation (warning: medpy 3D surface distance is slow)')
    parser.add_argument('--save_nii', action='store_true', default=False,
                        help='Save predicted 3D segmentation masks as .nii.gz')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save results')
    args = parser.parse_args()

    if not os.path.exists(args.weights_path):
        print(f"[Error] Checkpoint not found at: {args.weights_path}", flush=True)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.weights_path}", flush=True)
    print(f"Loading test volumes from: {args.volume_path}", flush=True)

    # Build model
    model = EMCADNet(num_classes=args.num_classes, encoder=args.encoder,
                     activation=args.activation_mscb, pretrain=False)

    checkpoint = torch.load(args.weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.cuda()
    model.eval()

    # Load dataset
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol",
                              list_dir=args.list_dir, nclass=args.num_classes)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    organ_names = [
        'Aorta', 'Gallbladder', 'Kidney (Left)', 'Kidney (Right)',
        'Liver', 'Pancreas', 'Spleen', 'Stomach'
    ]

    all_dices = []   # [num_cases, 8]
    all_hd95s = []   # [num_cases, 8]

    print(f"\nEvaluating {len(test_loader)} cases (HD95: {'ON' if args.eval_hd95 else 'OFF - Fast Dice Mode'})...", flush=True)
    start_total = time.time()

    for idx, sampled_batch in enumerate(test_loader):
        t0 = time.time()
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]

        case_d, case_h, pred_vol = evaluate_volume(
            image, label, model, classes=args.num_classes,
            patch_size=(args.img_size, args.img_size),
            eval_hd95=args.eval_hd95
        )
        all_dices.append(case_d)
        all_hd95s.append(case_h)
        dt = time.time() - t0

        case_mean_dice = np.mean(case_d) * 100.0
        print(f"[{idx+1:2d}/{len(test_loader)}] Case {case_name:<10} | Mean Dice: {case_mean_dice:6.2f}% | Elapsed: {dt:.1f}s", flush=True)

        if args.save_nii:
            try:
                import SimpleITK as sitk
                pred_itk = sitk.GetImageFromArray(pred_vol.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, os.path.join(args.output_dir, f"{case_name}_pred.nii.gz"))
            except Exception as e:
                print(f"Could not save .nii.gz for {case_name}: {e}", flush=True)

    total_time = time.time() - start_total
    all_dices = np.array(all_dices)  # [num_cases, 8]
    mean_dice_per_organ = np.mean(all_dices, axis=0) * 100.0
    overall_mean_dice = np.mean(mean_dice_per_organ)

    # Format result table
    if args.eval_hd95:
        all_hd95s = np.array(all_hd95s)
        mean_hd95_per_organ = np.mean(all_hd95s, axis=0)
        overall_mean_hd95 = np.mean(mean_hd95_per_organ)
        header = f"{'Organ':<18} | {'Dice (DSC %)':<14} | {'HD95 (mm)':<12}"
        divider = "-" * len(header)
        rows = [divider, header, divider]
        for i, organ in enumerate(organ_names):
            rows.append(f"{organ:<18} | {mean_dice_per_organ[i]:>12.2f}% | {mean_hd95_per_organ[i]:>10.2f} mm")
        rows.append(divider)
        rows.append(f"{'OVERALL MEAN':<18} | {overall_mean_dice:>12.2f}% | {overall_mean_hd95:>10.2f} mm")
        rows.append(divider)
    else:
        header = f"{'Organ':<18} | {'Dice (DSC %)':<14}"
        divider = "-" * len(header)
        rows = [divider, header, divider]
        for i, organ in enumerate(organ_names):
            rows.append(f"{organ:<18} | {mean_dice_per_organ[i]:>12.2f}%")
        rows.append(divider)
        rows.append(f"{'OVERALL MEAN':<18} | {overall_mean_dice:>12.2f}%")
        rows.append(divider)

    result_text = "\n".join(rows)
    print("\n" + result_text, flush=True)
    print(f"Total Evaluation Time: {total_time:.1f}s", flush=True)

    # Save to file
    summary_path = os.path.join(args.output_dir, "benchmark_results.txt")
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {args.weights_path}\n")
        f.write(f"Encoder: {args.encoder}\n\n")
        f.write(result_text + "\n")
    print(f"\n[Done] Results saved to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
