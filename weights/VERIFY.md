# Weights

Plain `state_dict`s (safetensors, 508 keys, the key set of `Nieuwlaar/ink9um-dense-native/weights/dense_native_016000.safetensors`).
Both are step 12 000 of the respective arm, selected by the rule fixed before training (see README).
The files are attached to the GitHub release of this repository (273 MB each, above the git file limit).

| file | sha256 |
| --- | --- |
| dense_native116_treatment_012000.safetensors | aae9c543a4d3dae2e639d277a1d7ae968bfb64bc123adb823caf241502004837 |
| dense_native116_control_012000.safetensors | 237174c60c89e0bc2d7423451792a64c2b164ccab583a840e4efb478765f36f9 |

Verify: `sha256sum weights/*.safetensors`

Load as a villa `ink_9um` checkpoint (same as Nieuwlaar's README):

```python
import json, torch
from safetensors.torch import load_file
torch.save({"model": load_file("dense_native116_treatment_012000.safetensors"),
            "config": json.load(open("configs/train_treatment.json")), "step": 12000},
           "dense_native116_treatment_012000.pth")
```
