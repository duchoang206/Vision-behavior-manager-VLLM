// DeepStream instance-mask parser for RAW Ultralytics YOLOv8/YOLO11-seg ONNX exports (nms=False).
//
// The stock DeepStream-Yolo-Seg parser (NvDsInferParseYoloSeg) expects a pre-processed tensor
// [N, 6 + mask] produced by their exporter (EfficientNMS + ROIAlign TensorRT plugins). A plain Ultralytics
// export has two outputs instead:
//   output0 [4 + classes + mask_coeffs, anchors]   (cx, cy, w, h, class scores..., coefficients...)
//   output1 [mask_coeffs, proto_h, proto_w]        (prototype masks)
// Feeding that to the stock parser makes it read the wrong axis and emit no valid objects. This parser decodes
// the raw tensors itself: threshold per class, class-wise NMS, then coefficient x prototype masks resampled into
// a bbox-relative MASK_SIZE x MASK_SIZE bitmap, which is what NvDsObjectMeta.mask_params expects.
//
// Build: g++ -std=c++17 -shared -fPIC -O2 -I/opt/nvidia/deepstream/deepstream/sources/includes \
//        yolo_seg_raw_parser.cpp -o libnvdsinfer_custom_impl_Yolo_seg_raw.so

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYoloSegRaw(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList);

namespace {

constexpr int MASK_SIZE = 160;      // bbox-relative mask resolution handed to DeepStream
constexpr float NMS_IOU = 0.45f;
constexpr size_t MAX_CANDIDATES = 300;
constexpr size_t MAX_OBJECTS = 100;

struct Candidate {
  float x1, y1, x2, y2, score;
  int classId;
  size_t anchor;
};

float iou(const Candidate& a, const Candidate& b) {
  const float left = std::max(a.x1, b.x1), top = std::max(a.y1, b.y1);
  const float right = std::min(a.x2, b.x2), bottom = std::min(a.y2, b.y2);
  const float inter = std::max(0.f, right - left) * std::max(0.f, bottom - top);
  const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

float clampf(float v, float lo, float hi) { return std::min(hi, std::max(lo, v)); }

}  // namespace

extern "C" bool NvDsInferParseYoloSegRaw(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList) {
  const NvDsInferLayerInfo* detections = nullptr;
  const NvDsInferLayerInfo* protos = nullptr;
  for (const auto& layer : outputLayersInfo) {
    if (layer.inferDims.numDims == 2 && !detections) detections = &layer;
    else if (layer.inferDims.numDims == 3 && !protos) protos = &layer;
  }
  if (!detections || !protos) {
    std::cerr << "ERROR - raw YOLO-seg parser needs a 2-D detection output and a 3-D prototype output" << std::endl;
    return false;
  }

  const int classes = static_cast<int>(detectionParams.numClassesConfigured);
  const size_t channels = detections->inferDims.d[0];
  const size_t anchors = detections->inferDims.d[1];
  const int coeffs = static_cast<int>(channels) - 4 - classes;
  const int protoCount = protos->inferDims.d[0];
  const int protoH = protos->inferDims.d[1];
  const int protoW = protos->inferDims.d[2];
  if (coeffs <= 0 || coeffs != protoCount) {
    std::cerr << "ERROR - raw YOLO-seg parser: " << channels << " channels do not match " << classes
              << " classes + " << protoCount << " mask coefficients" << std::endl;
    return false;
  }

  const float* out = static_cast<const float*>(detections->buffer);
  const float* proto = static_cast<const float*>(protos->buffer);
  const float netW = static_cast<float>(networkInfo.width), netH = static_cast<float>(networkInfo.height);

  std::vector<Candidate> candidates;
  for (size_t a = 0; a < anchors; ++a) {
    int best = 0;
    float bestScore = out[4 * anchors + a];
    for (int c = 1; c < classes; ++c) {
      const float s = out[(4 + c) * anchors + a];
      if (s > bestScore) { bestScore = s; best = c; }
    }
    const float threshold = best < static_cast<int>(detectionParams.perClassPreclusterThreshold.size())
        ? detectionParams.perClassPreclusterThreshold[best] : 0.25f;
    if (bestScore < threshold) continue;
    const float cx = out[0 * anchors + a], cy = out[1 * anchors + a], w = out[2 * anchors + a], h = out[3 * anchors + a];
    Candidate cand{clampf(cx - w / 2, 0, netW), clampf(cy - h / 2, 0, netH), clampf(cx + w / 2, 0, netW),
                   clampf(cy + h / 2, 0, netH), bestScore, best, a};
    if (cand.x2 - cand.x1 < 1.f || cand.y2 - cand.y1 < 1.f) continue;
    candidates.push_back(cand);
  }
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& l, const Candidate& r) { return l.score > r.score; });
  if (candidates.size() > MAX_CANDIDATES) candidates.resize(MAX_CANDIDATES);

  std::vector<Candidate> kept;
  for (const auto& cand : candidates) {
    bool suppressed = false;
    for (const auto& other : kept) {
      if (other.classId == cand.classId && iou(cand, other) > NMS_IOU) { suppressed = true; break; }
    }
    if (!suppressed) kept.push_back(cand);
    if (kept.size() >= MAX_OBJECTS) break;
  }

  const float scaleX = protoW / netW, scaleY = protoH / netH;
  std::vector<float> logits(static_cast<size_t>(protoW) * protoH);
  objectList.clear();
  for (const auto& cand : kept) {
    std::vector<float> coef(coeffs);
    for (int k = 0; k < coeffs; ++k) coef[k] = out[(4 + classes + k) * anchors + cand.anchor];

    // Coefficient-weighted prototype logits, only inside the box (in prototype space).
    const int px1 = std::max(0, static_cast<int>(std::floor(cand.x1 * scaleX)));
    const int py1 = std::max(0, static_cast<int>(std::floor(cand.y1 * scaleY)));
    const int px2 = std::min(protoW - 1, static_cast<int>(std::ceil(cand.x2 * scaleX)));
    const int py2 = std::min(protoH - 1, static_cast<int>(std::ceil(cand.y2 * scaleY)));
    for (int y = py1; y <= py2; ++y) {
      for (int x = px1; x <= px2; ++x) {
        float sum = 0.f;
        const size_t offset = static_cast<size_t>(y) * protoW + x;
        for (int k = 0; k < coeffs; ++k) sum += coef[k] * proto[static_cast<size_t>(k) * protoH * protoW + offset];
        logits[offset] = sum;
      }
    }

    NvDsInferInstanceMaskInfo b;
    std::memset(&b, 0, sizeof(b));
    b.left = cand.x1;
    b.top = cand.y1;
    b.width = cand.x2 - cand.x1;
    b.height = cand.y2 - cand.y1;
    b.classId = cand.classId;
    b.detectionConfidence = cand.score;
    b.mask_width = MASK_SIZE;
    b.mask_height = MASK_SIZE;
    b.mask_size = sizeof(float) * MASK_SIZE * MASK_SIZE;
    b.mask = new float[MASK_SIZE * MASK_SIZE];
    for (int i = 0; i < MASK_SIZE; ++i) {
      const float fy = (cand.y1 + (i + 0.5f) * b.height / MASK_SIZE) * scaleY - 0.5f;
      const int y0 = std::min(py2, std::max(py1, static_cast<int>(std::floor(fy))));
      const int y1 = std::min(py2, y0 + 1);
      const float wy = clampf(fy - y0, 0.f, 1.f);
      for (int j = 0; j < MASK_SIZE; ++j) {
        const float fx = (cand.x1 + (j + 0.5f) * b.width / MASK_SIZE) * scaleX - 0.5f;
        const int x0 = std::min(px2, std::max(px1, static_cast<int>(std::floor(fx))));
        const int x1 = std::min(px2, x0 + 1);
        const float wx = clampf(fx - x0, 0.f, 1.f);
        const float top = logits[static_cast<size_t>(y0) * protoW + x0] * (1 - wx) + logits[static_cast<size_t>(y0) * protoW + x1] * wx;
        const float bottom = logits[static_cast<size_t>(y1) * protoW + x0] * (1 - wx) + logits[static_cast<size_t>(y1) * protoW + x1] * wx;
        const float logit = top * (1 - wy) + bottom * wy;
        b.mask[i * MASK_SIZE + j] = 1.f / (1.f + std::exp(-logit));
      }
    }
    objectList.push_back(b);
  }
  return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloSegRaw);
