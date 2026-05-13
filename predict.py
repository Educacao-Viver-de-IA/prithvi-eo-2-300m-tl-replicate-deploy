import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/src/hf-cache"

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

WEIGHTS_DIR = "/src/hf-cache"
EXPECTED_BANDS = 6  # Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] WEIGHTS_DIR={WEIGHTS_DIR}", flush=True)
        try:
            print(f"[setup] dir contents: {sorted(os.listdir(WEIGHTS_DIR))[:20]}", flush=True)
        except Exception as e:
            print(f"[setup] cannot list WEIGHTS_DIR: {e}", flush=True)
        print(f"[setup] cuda: {torch.cuda.is_available()}", flush=True)

        print(f"[setup] importing terratorch... (t={time.time()-t0:.1f}s)", flush=True)
        from terratorch.registry import BACKBONE_REGISTRY
        self.BACKBONE_REGISTRY = BACKBONE_REGISTRY

        print(f"[setup] building prithvi_eo_v2_300_tl... (t={time.time()-t0:.1f}s)", flush=True)
        # Backbone com 300M params, TL (temporal/location embeddings)
        self.model = BACKBONE_REGISTRY.build(
            "prithvi_eo_v2_300_tl",
            pretrained=True,
        )
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda().to(torch.float32)
        print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)

    def _load_image_as_6band(self, image_path: Path, image_size: int) -> torch.Tensor:
        """
        Carrega imagem como tensor [B=1, C=6, T=1, H, W] (formato Prithvi).
        Aceita:
        - GeoTIFF multissensorial (6 bandas) — uso canônico
        - GeoTIFF qualquer Nº bandas — pad/replicate até 6
        - PNG/JPG RGB — replica pra 6 bandas (aproximação pra demo)
        """
        path_str = str(image_path)
        arr = None

        if path_str.lower().endswith((".tif", ".tiff")):
            try:
                import rasterio
                with rasterio.open(path_str) as src:
                    arr = src.read().astype(np.float32)  # [bands, H, W]
                    print(f"[predict] rasterio: {arr.shape} bandas detectadas", flush=True)
            except Exception as e:
                print(f"[predict] rasterio falhou ({e}), fallback PIL", flush=True)

        if arr is None:
            pil = Image.open(image_path)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            arr = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)  # [3, H, W]
            print(f"[predict] PIL: {arr.shape}", flush=True)

        n_bands = arr.shape[0]
        if n_bands < EXPECTED_BANDS:
            # Pad replicando o último canal
            pad_size = EXPECTED_BANDS - n_bands
            pad = np.repeat(arr[-1:], pad_size, axis=0)
            arr = np.concatenate([arr, pad], axis=0)
            print(f"[predict] padded de {n_bands} pra {EXPECTED_BANDS} bandas", flush=True)
        elif n_bands > EXPECTED_BANDS:
            arr = arr[:EXPECTED_BANDS]
            print(f"[predict] cortado de {n_bands} pra {EXPECTED_BANDS} bandas", flush=True)

        # Normalizar: valores grandes (uint16) -> [0, 1]; valores pequenos (uint8) -> [0, 1]
        max_v = arr.max()
        if max_v > 255:
            arr = arr / 10000.0  # HLS reflectance scale (0-10000)
        elif max_v > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        # Resize com PIL pra cada banda
        import torch.nn.functional as F
        tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, 6, H, W]
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
        # Adicionar dimensão temporal T=1: [B, C, T, H, W]
        tensor = tensor.unsqueeze(2)
        return tensor

    def predict(
        self,
        image: Path = Input(
            description="Imagem multissensorial (GeoTIFF 6 bandas HLS preferencial: B/G/R/NIR/SWIR1/SWIR2). RGB também aceito (auto-pad pra 6 bandas).",
        ),
        image_size: int = Input(
            description="Resolução de entrada (múltiplo de patch size = 16). Use 224 ou 512.",
            default=224,
            ge=64,
            le=1024,
        ),
        latitude: float = Input(
            description="(Opcional, TL feature) Latitude do centro da imagem (-90 a 90). Use 0 se não souber.",
            default=0.0,
            ge=-90.0,
            le=90.0,
        ),
        longitude: float = Input(
            description="(Opcional, TL feature) Longitude do centro (-180 a 180). Use 0 se não souber.",
            default=0.0,
            ge=-180.0,
            le=180.0,
        ),
        year: int = Input(
            description="(Opcional, TL feature) Ano de aquisição da imagem (ex: 2024).",
            default=2024,
            ge=1980,
            le=2100,
        ),
        day_of_year: int = Input(
            description="(Opcional, TL feature) Dia do ano (1-365).",
            default=180,
            ge=1,
            le=366,
        ),
        return_format: str = Input(
            description="'summary' (estatísticas + preview) ou 'full' (embedding completo).",
            default="summary",
            choices=["summary", "full"],
        ),
    ) -> str:
        device = next(self.model.parameters()).device

        tensor = self._load_image_as_6band(image, image_size).to(device, dtype=torch.float32)
        print(f"[predict] input shape: {tensor.shape}", flush=True)

        # Temporal coords: 3D [batch, time_steps, 2] — terratorch indexa temporal_coords[:, :, 0]
        # Location coords: 2D [batch, 2] — terratorch indexa location_coords[:, 1]
        # (descobertos por tentativa/erro — o shape difere entre os dois apesar de parecidos)
        temporal = torch.tensor([[[year, day_of_year]]], dtype=torch.float32, device=device)  # [1, 1, 2]
        location = torch.tensor([[latitude, longitude]], dtype=torch.float32, device=device)  # [1, 2]

        with torch.no_grad():
            # Tentativa 1: passa kwargs temporal/location
            try:
                output = self.model(tensor, temporal_coords=temporal, location_coords=location)
            except TypeError:
                try:
                    output = self.model(tensor, time=temporal, location=location)
                except TypeError:
                    # Fallback: sem metadados
                    output = self.model(tensor)

        # Normalizar output
        if isinstance(output, (list, tuple)):
            feat = output[-1]  # último layer geralmente é o output
        elif isinstance(output, dict):
            feat = next(iter(output.values()))
        else:
            feat = output

        # [B, N_tokens, dim] -> mean pool -> [B, dim]
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        feat_vec = feat.squeeze(0).cpu().numpy()

        result = {
            "model": "prithvi-eo-2-300m-tl",
            "embedding_dim": int(feat_vec.shape[-1]) if feat_vec.ndim >= 1 else 0,
            "image_size": image_size,
            "metadata": {
                "latitude": latitude,
                "longitude": longitude,
                "year": year,
                "day_of_year": day_of_year,
            },
        }

        if return_format == "full":
            result["embedding"] = feat_vec.flatten().tolist()
        else:
            result["mean"] = float(feat_vec.mean())
            result["std"] = float(feat_vec.std())
            result["min"] = float(feat_vec.min())
            result["max"] = float(feat_vec.max())
            result["norm"] = float(np.linalg.norm(feat_vec))
            result["embedding_preview"] = feat_vec.flatten()[:32].tolist()

        return json.dumps(result, ensure_ascii=False)
