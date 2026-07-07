from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import traceback

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


# ViewPosition / projection values on which CTR is NOT reliable (heart magnified).
_UNRELIABLE_VIEWS = {"AP", "AP SUPINE", "SUPINE", "PORTABLE", "SEMI-ERECT", "SEMI ERECT", "LL", "RL"}


class CardiothoracicRatioInput(BaseModel):
    """Input schema for the Cardiothoracic Ratio (CTR) Tool."""

    image_path: str = Field(
        ..., description="Path to the frontal chest X-ray image file (PA view preferred)"
    )
    dicom_path: Optional[str] = Field(
        None,
        description="Optional path to the source DICOM file. If given, the projection "
        "(ViewPosition) is read to gate reliability, and PixelSpacing is used to also report "
        "the diameters in centimetres when the PNG matches the DICOM dimensions.",
    )


class CardiothoracicRatioTool(BaseTool):
    """Tool for computing the cardiothoracic ratio (CTR) from a frontal chest X-ray.

    CTR = maximal transverse cardiac diameter / maximal transverse (inner) thoracic diameter.
    The cardiac diameter is taken as the maximum horizontal extent of the heart segmentation;
    the thoracic diameter as the maximum horizontal extent of the combined lung fields
    (an approximation of the inner rib-cage width). Because CTR is a ratio of two widths in
    the same image, it is independent of pixel spacing and does not require DICOM calibration.

    When a DICOM path is supplied the projection is checked (AP/portable films magnify the
    heart and are flagged), and when an optional classifier is injected its Cardiomegaly
    probability is used as an independent cross-check.
    """

    name: str = "cardiothoracic_ratio"
    description: str = (
        "Computes the cardiothoracic ratio (CTR) on a frontal chest X-ray by segmenting the "
        "heart and lungs. Prefer this tool over raw segmentation metrics for any 'is the heart "
        "enlarged / cardiomegaly / measure CTR' question. Returns the CTR value, the cardiac and "
        "thoracic transverse diameters (pixels, and cm when a DICOM is provided), a category "
        "(normal < 0.5, borderline 0.5-0.55, enlarged > 0.55), a confidence score, an annotated "
        "overlay, and warnings. CTR is only valid on a PA (frontal, upright) view; pass the "
        "optional dicom_path so the tool can detect AP/portable projections and downgrade "
        "reliability automatically. Being a ratio, the value itself does not depend on pixel "
        "spacing and works on PNG/JPG without DICOM calibration."
    )
    args_schema: Type[BaseModel] = CardiothoracicRatioInput

    model: Any = None
    device: Optional[str] = "cuda"
    transform: Any = None
    temp_dir: Path = Path("temp")
    # Optional injected ChestXRayClassifierTool used for an independent Cardiomegaly cross-check.
    # Injected after construction (see main.py) so no second model is loaded onto the GPU.
    classifier: Any = None

    # Target indices in the ChestX-Det PSPNet output.
    HEART_IDX: int = 8
    LEFT_LUNG_IDX: int = 4
    RIGHT_LUNG_IDX: int = 5

    def __init__(self, device: Optional[str] = "cuda", temp_dir: Optional[Path] = Path("temp")):
        """Initialize the CTR tool with the ChestX-Det segmentation model."""
        super().__init__()
        self.model = xrv.baseline_models.chestx_det.PSPNet()
        self.device = torch.device(device) if device else "cuda"
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)]
        )

        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    def _align_mask_to_original(
        self, mask: np.ndarray, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Align a mask from the transformed (cropped/resized) space back to the full original image.
        Assumes the transform center-crops to a square of side = min(height, width) and resizes
        to (512, 512). This mirrors ChestXRaySegmentationTool so the two tools stay consistent.
        """
        orig_h, orig_w = original_shape
        crop_size = min(orig_h, orig_w)
        crop_top = (orig_h - crop_size) // 2
        crop_left = (orig_w - crop_size) // 2

        resized_mask = skimage.transform.resize(
            mask, (crop_size, crop_size), order=0, preserve_range=True, anti_aliasing=False
        )
        full_mask = np.zeros(original_shape)
        full_mask[crop_top : crop_top + crop_size, crop_left : crop_left + crop_size] = resized_mask
        return full_mask

    def _max_horizontal_extent(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Find the row of maximum horizontal extent of a binary mask.

        Returns (width, row, left_col, right_col). Width is 0 if the mask is empty.
        Scanning per-row (rather than using a bounding box) gives the true maximal transverse
        diameter, which is how CTR diameters are defined clinically.
        """
        rows = np.where(mask.any(axis=1))[0]
        if len(rows) == 0:
            return 0, 0, 0, 0

        best_width = 0
        best_row = int(rows[0])
        best_left = 0
        best_right = 0
        for r in rows:
            cols = np.where(mask[r])[0]
            left, right = int(cols[0]), int(cols[-1])
            width = right - left + 1
            if width > best_width:
                best_width = width
                best_row = int(r)
                best_left = left
                best_right = right
        return best_width, best_row, best_left, best_right

    @staticmethod
    def _categorize(ctr: float) -> str:
        """Map a CTR value to a clinical category (PA view convention)."""
        if ctr < 0.5:
            return "normal"
        if ctr <= 0.55:
            return "borderline"
        return "enlarged"

    def _read_dicom_view(self, dicom_path: str) -> Dict[str, Any]:
        """Read projection and calibration hints from a DICOM file. Best-effort; never raises."""
        info: Dict[str, Any] = {"view": None, "view_reliable": None, "pixel_spacing": None,
                                "rows": None, "columns": None}
        try:
            import pydicom

            dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
            view = getattr(dcm, "ViewPosition", None)
            if view:
                view = str(view).strip().upper()
                info["view"] = view
                info["view_reliable"] = view == "PA"
            spacing = getattr(dcm, "ImagerPixelSpacing", None) or getattr(dcm, "PixelSpacing", None)
            if spacing is not None:
                info["pixel_spacing"] = [float(s) for s in spacing]
            info["rows"] = getattr(dcm, "Rows", None)
            info["columns"] = getattr(dcm, "Columns", None)
        except Exception:
            pass
        return info

    def _cross_check_classifier(self, image_path: str, ctr: float) -> Dict[str, Any]:
        """Use an injected classifier's Cardiomegaly probability as an independent check."""
        result: Dict[str, Any] = {
            "cardiomegaly_prob": None,
            "enlarged_cardiomediastinum_prob": None,
            "agreement": "unavailable",
        }
        if self.classifier is None:
            return result
        try:
            cls_out, _ = self.classifier._run(image_path)
            if "error" in cls_out:
                return result
            cardio = cls_out.get("Cardiomegaly")
            enl = cls_out.get("Enlarged Cardiomediastinum")
            result["cardiomegaly_prob"] = round(float(cardio), 3) if cardio is not None else None
            result["enlarged_cardiomediastinum_prob"] = (
                round(float(enl), 3) if enl is not None else None
            )
            if cardio is not None:
                ctr_says_big = ctr >= 0.5
                cls_says_big = float(cardio) >= 0.5
                result["agreement"] = "concordant" if ctr_says_big == cls_says_big else "discordant"
        except Exception:
            pass
        return result

    def _save_visualization(
        self,
        original_img: np.ndarray,
        heart_line: Tuple[int, int, int],
        thorax_line: Tuple[int, int, int],
        ctr: float,
    ) -> str:
        """Save an overlay of the cardiac and thoracic measurement lines."""
        plt.figure(figsize=(10, 10))
        plt.imshow(original_img, cmap="gray")

        h_row, h_left, h_right = heart_line
        t_row, t_left, t_right = thorax_line

        plt.hlines(t_row, t_left, t_right, colors="cyan", linewidth=2.5, label="Thoracic diameter")
        plt.hlines(h_row, h_left, h_right, colors="red", linewidth=2.5, label="Cardiac diameter")

        plt.title(f"Cardiothoracic Ratio = {ctr:.2f} ({self._categorize(ctr)})")
        plt.legend(loc="lower right")
        plt.axis("off")

        save_path = self.temp_dir / f"ctr_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        return str(save_path)

    def _run(
        self,
        image_path: str,
        dicom_path: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Compute the cardiothoracic ratio for a frontal chest X-ray."""
        try:
            # Strip stray quotes/space some local models add around path arguments.
            image_path = image_path.strip().strip("\"'").strip()
            if dicom_path:
                dicom_path = dicom_path.strip().strip("\"'").strip()

            # Load and process image (same pipeline as the segmentation tool).
            original_img = skimage.io.imread(image_path)
            if len(original_img.shape) > 2:
                original_img = original_img[:, :, 0]

            img = xrv.datasets.normalize(original_img, 255)
            img = img[None, ...]
            img = self.transform(img)
            img = torch.from_numpy(img)
            img = img.to(self.device)

            with torch.no_grad():
                pred = self.model(img)
            pred_probs = torch.sigmoid(pred)
            pred_masks = (pred_probs > 0.5).float()

            heart_mask = pred_masks[0, self.HEART_IDX].cpu().numpy()
            left_lung = pred_masks[0, self.LEFT_LUNG_IDX].cpu().numpy()
            right_lung = pred_masks[0, self.RIGHT_LUNG_IDX].cpu().numpy()
            lung_mask = np.logical_or(left_lung > 0, right_lung > 0).astype(float)

            if heart_mask.sum() == 0 or lung_mask.sum() == 0:
                missing = "heart" if heart_mask.sum() == 0 else "lungs"
                raise ValueError(
                    f"Could not segment the {missing}; the image may not be a frontal chest "
                    "X-ray or the anatomy is not clearly visible."
                )

            # Segmentation confidence: mean model probability within each mask (512-space).
            heart_conf = float(pred_probs[0, self.HEART_IDX].cpu().numpy()[heart_mask > 0].mean())
            lung_prob = torch.maximum(
                pred_probs[0, self.LEFT_LUNG_IDX], pred_probs[0, self.RIGHT_LUNG_IDX]
            ).cpu().numpy()
            lung_conf = float(lung_prob[lung_mask > 0].mean())
            seg_confidence = (heart_conf + lung_conf) / 2.0

            # Align masks back to full-resolution image coordinates before measuring, so the
            # center-crop does not silently shrink the measured widths.
            heart_full = self._align_mask_to_original(heart_mask, original_img.shape)
            lung_full = self._align_mask_to_original(lung_mask, original_img.shape)

            cardiac_w, h_row, h_left, h_right = self._max_horizontal_extent(heart_full > 0)
            thoracic_w, t_row, t_left, t_right = self._max_horizontal_extent(lung_full > 0)

            if thoracic_w == 0:
                raise ValueError("Thoracic diameter measured as zero; cannot compute CTR.")

            ctr = cardiac_w / thoracic_w
            category = self._categorize(ctr)

            # Warn if the lung fields touch the lateral edges of the center-cropped region,
            # which means the true thoracic width may have been clipped (inflating the CTR).
            crop_size = min(original_img.shape)
            crop_left = (original_img.shape[1] - crop_size) // 2
            crop_right = crop_left + crop_size - 1
            edge_margin = max(2, int(0.01 * crop_size))
            clipped = t_left <= crop_left + edge_margin or t_right >= crop_right - edge_margin

            # --- View gating from DICOM (optional) ---
            dicom_info = self._read_dicom_view(dicom_path) if dicom_path else {}
            view = dicom_info.get("view")
            view_reliable = dicom_info.get("view_reliable")

            warnings: List[str] = []
            if view_reliable is True:
                pass  # confirmed PA — no projection caveat needed
            elif view is not None and view in _UNRELIABLE_VIEWS:
                warnings.append(
                    f"Projection is '{view}' (not PA): the heart is magnified on this view, so the "
                    "CTR is overestimated and the < 0.5 threshold does not apply. Treat as unreliable."
                )
            else:
                warnings.append(
                    "Projection could not be confirmed as PA. CTR is only clinically valid on a PA "
                    "(frontal, upright) view; on AP/portable films the ratio is overestimated. "
                    "Pass dicom_path to detect the projection automatically."
                )
            if clipped:
                warnings.append(
                    "The lung fields reach the edge of the analyzed (center-cropped) region, so the "
                    "thoracic diameter may be underestimated and the CTR overestimated."
                )

            # --- Classifier cross-check (optional) ---
            cross = self._cross_check_classifier(image_path, ctr)
            if cross["agreement"] == "discordant":
                warnings.append(
                    f"Independent classifier disagrees: Cardiomegaly probability "
                    f"{cross['cardiomegaly_prob']} vs CTR {round(ctr, 3)} "
                    f"({'enlarged' if ctr >= 0.5 else 'not enlarged'}). Interpret with caution."
                )

            # --- Confidence: segmentation quality, penalized by known risk factors ---
            confidence = seg_confidence
            if view_reliable is False or (view is not None and view in _UNRELIABLE_VIEWS):
                confidence *= 0.6
            elif view_reliable is None:
                confidence *= 0.9  # projection unconfirmed
            if clipped:
                confidence *= 0.75
            if cross["agreement"] == "discordant":
                confidence *= 0.8
            confidence = round(min(max(confidence, 0.0), 1.0), 3)

            # --- Optional centimetre calibration when DICOM matches the PNG dimensions ---
            cardiac_cm = thoracic_cm = None
            spacing = dicom_info.get("pixel_spacing")
            rows, cols = dicom_info.get("rows"), dicom_info.get("columns")
            if spacing and rows and cols and (int(rows), int(cols)) == tuple(original_img.shape):
                col_spacing_mm = float(spacing[1]) if len(spacing) > 1 else float(spacing[0])
                cardiac_cm = round(cardiac_w * col_spacing_mm / 10.0, 1)
                thoracic_cm = round(thoracic_w * col_spacing_mm / 10.0, 1)
            elif spacing:
                warnings.append(
                    "PixelSpacing was present but the PNG dimensions do not match the DICOM; "
                    "centimetre calibration was skipped (the CTR ratio is unaffected)."
                )

            viz_path = self._save_visualization(
                original_img, (h_row, h_left, h_right), (t_row, t_left, t_right), ctr
            )

            output = {
                "cardiothoracic_ratio": round(float(ctr), 3),
                "category": category,
                "confidence": confidence,
                "view": view,
                "view_reliable": view_reliable,
                "cardiac_diameter_pixels": int(cardiac_w),
                "thoracic_diameter_pixels": int(thoracic_w),
                "cardiac_diameter_cm": cardiac_cm,
                "thoracic_diameter_cm": thoracic_cm,
                "classifier_cross_check": cross,
                "ctr_image_path": viz_path,
                "warnings": warnings,
            }

            metadata = {
                "image_path": image_path,
                "dicom_path": dicom_path,
                "ctr_image_path": viz_path,
                "original_size": original_img.shape,
                "model_size": tuple(img.shape[-2:]),
                "segmentation_confidence": round(seg_confidence, 3),
                "cardiac_line": {"row": h_row, "left": h_left, "right": h_right},
                "thoracic_line": {"row": t_row, "left": t_left, "right": t_right},
                "thoracic_possibly_clipped": bool(clipped),
                "analysis_status": "completed",
            }

            return output, metadata

        except Exception as e:
            error_output = {"error": str(e)}
            error_metadata = {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_traceback": traceback.format_exc(),
            }
            return error_output, error_metadata

    async def _arun(
        self,
        image_path: str,
        dicom_path: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, dicom_path)
