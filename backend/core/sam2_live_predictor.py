"""Commit only identity-verified live observations to bounded SAM2 memory."""

import torch
from ultralytics.models.sam import SAM2DynamicInteractivePredictor


class LiveSAM2Predictor(SAM2DynamicInteractivePredictor):
    @torch.inference_mode()
    def seed_reference_mask(self, mask):
        features = self.vision_feats[-1].permute(1, 2, 0).reshape(1, -1, *self.feat_sizes[-1])
        _, _, _, low, high, pointer, score = self.model._use_mask_as_output(
            mask, backbone_features=features, high_res_features=self.high_res_features)
        self.live_output = dict(pred_masks=low.to(features.dtype), pred_masks_high_res=high.to(features.dtype),
                                obj_ptr=pointer.to(features.dtype), object_score_logits=score.to(features.dtype))
        self.obj_idx_set.add(0)
        self.commit_live_memory(limit=1)

    @torch.inference_mode()
    def infer_live(self, image, box=None):
        self.get_im_features(image)
        if box is not None:
            points, labels, _ = self._prepare_prompts(dst_shape=self.imgsz, src_shape=self.src_shape, bboxes=box)
            output = self.track_step(obj_idx=0, point=points[:1], label=labels[:1])
            self.obj_idx_set.add(0)
        else:
            if not self.memory_bank:
                raise RuntimeError("Live SAM2 requires a verified prompt before propagation.")
            output = self.track_step()
        self.live_output = output
        scores = (output["object_score_logits"] / 32).clamp(min=0)
        return output["pred_masks"].flatten(0, 1), scores.flatten(0, 1)

    def track_step(self, obj_idx=None, point=None, label=None, mask=None):
        output = super().track_step(obj_idx, point, label, mask)
        if obj_idx is None:
            self.live_output = output
        return output

    @torch.inference_mode()
    def commit_live_memory(self, limit=3, reference_count=1):
        output = getattr(self, "live_output", None)
        if output is None:
            return False
        features, positions = self.model._encode_new_memory(
            current_vision_feats=self.vision_feats,
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=output["pred_masks_high_res"],
            object_score_logits=output["object_score_logits"],
            is_mask_from_pts=False,
        )
        memory = {key: output[key].detach() for key in ("pred_masks", "obj_ptr", "object_score_logits")}
        memory.update(maskmem_features=features.detach(), maskmem_pos_enc=positions)
        limit = max(1, min(int(limit), self.model.num_maskmem))
        self.memory_bank.append(memory)
        if limit == 1:
            self.memory_bank[:] = self.memory_bank[-1:]
        elif len(self.memory_bank) > limit:
            keep = min(max(0, reference_count), limit - 1)
            self.memory_bank[:] = self.memory_bank[:keep] + self.memory_bank[-(limit - keep):]
        self.live_output = None
        return True
