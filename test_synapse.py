import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import zoom
try:
    from medpy import metric
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False

from lib.networks import EMCADNet
from utils.dataset_synapse import Synapse_dataset


def calculate_metrics_for_case(pred, gt):
    """
    Calculate Dice Similarity Coefficient (DSC) and 95% Hausdorff Distance (HD95)
    for a binary prediction and ground truth pair.
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        if HAS_MEDPY:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
        else:
            intersection = np.sum(pred * gt)
            dice = (2.0 * intersection) / (np.sum(pred) + np.sum(gt))
            hd95 = 0.0
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1.0, 0.0
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0.0, 50.0  # standard penalty distance when structure is missed
    else:
        return 1.0, 0.0


def evaluate_volume(image, label, net, classes, patch_size=(224, 224)):
    """
    Slice-by-slice 3D volume inference.
    Takes 3D volume [Depth, H, W], resizes slices to patch_size, runs model,
    resizes back to original slice dimension, and evaluates per-class metrics.
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
                if isinstance(P, list):
                    outputs = P[-1]  # Finest resolution prediction
                else:
                    outputs = P
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

    case_metrics = []
    for c in range(1, classes):
        dice, hd95 = calculate_metrics_for_case(prediction == c, label == c)
        case_metrics.append((dice, hd95))

    return case_metrics, prediction


def main():
    parser = argparse.ArgumentParser(description="Test Synapse segmentation with best.pth")
    parser.add_argument('--volume_path', type=str, default='./data/synapse/test_vol_h5_new',
                        help='Root directory for test volume .npy.h5 data')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse',
                        help='Directory containing test_vol.txt')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the trained checkpoint (e.g., model_pth/.../best.pth)')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of classes including background (default 9)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image resolution (default 224)')
    parser.add_argument('--encoder', type=str, default='pvt_v2_b2',
                        help='Encoder architecture: pvt_v2_b2, resnet34, etc.')
    parser.add_argument('--activation_mscb', type=str, default='relu6',
                        help='Activation used in MSCB: relu6 or relu (default: relu6)')
    parser.add_argument('--save_nii', action='store_true', default=False,
                        help='Save predicted 3D segmentation masks as .nii.gz')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Directory to save results and masks')
    args = parser.parse_args()

    if not os.path.exists(args.weights_path):
        print(f"[Error] Checkpoint not found at: {args.weights_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint: {args.weights_path}")
    print(f"Loading test volumes from: {args.volume_path}")

    # Build model (pretrain=False since we load our own checkpoint)
    model = EMCADNet(num_classes=args.num_classes, encoder=args.encoder, activation=args.activation_mscb, pretrain=False)

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

    all_case_metrics = []  # shape: [num_cases, num_organs, 2] (dice, hd95)

    print(f"\nStarting evaluation on {len(test_loader)} test cases...")
    for sampled_batch in tqdm(test_loader, desc="Evaluating"):
        image = sampled_batch["image"]
        label = sampled_batch["label"]
        case_name = sampled_batch['case_name'][0]

        case_metrics, pred_vol = evaluate_volume(
            image, label, model, classes=args.num_classes,
            patch_size=(args.img_size, args.img_size)
        )
        all_case_metrics.append(case_metrics)

        # Optional: Save .nii.gz segmentation mask
        if args.save_nii:
            try:
                import SimpleITK as sitk
                pred_itk = sitk.GetImageFromArray(pred_vol.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, os.path.join(args.output_dir, f"{case_name}_pred.nii.gz"))
            except Exception as e:
                print(f"Could not save .nii.gz for {case_name}: {e}")

    # Compute average metrics across all cases
    all_case_metrics = np.array(all_case_metrics)  # [num_cases, 8, 2]
    mean_per_organ = np.mean(all_case_metrics, axis=0)  # [8, 2]
    overall_mean = np.mean(mean_per_organ, axis=0)  # [2]

    # Format result table
    header = f"{'Organ':<18} | {'Dice (DSC %)':<14} | {'HD95 (mm)':<12}"
    divider = "-" * len(header)
    rows = [divider, header, divider]

    for i, organ in enumerate(organ_names):
        dice_val = mean_per_organ[i, 0] * 100.0
        hd95_val = mean_per_organ[i, 1]
        rows.append(f"{organ:<18} | {dice_val:>12.2f}% | {hd95_val:>10.2f} mm")

    rows.append(divider)
    mean_dice = overall_mean[0] * 100.0
    mean_hd95 = overall_mean[1]
    rows.append(f"{'OVERALL MEAN':<18} | {mean_dice:>12.2f}% | {mean_hd95:>10.2f} mm")
    rows.append(divider)

    result_text = "\n".join(rows)
    print("\n" + result_text)

    # Save results to text file
    summary_path = os.path.join(args.output_dir, "benchmark_results.txt")
    with open(summary_path, "w") as f:
        f.write(f"Checkpoint: {args.weights_path}\n")
        f.write(f"Encoder: {args.encoder}\n\n")
        f.write(result_text + "\n")
    print(f"\n[Done] Results saved to: {summary_path}")


if __name__ == "__main__":
    main()
